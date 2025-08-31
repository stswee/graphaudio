import os
import argparse
import csv
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity

def _path_tag(path_str: str) -> str:
    base = os.path.basename(os.path.normpath(path_str))
    if base.lower().endswith(".csv"):
        base = base[:-4]
    return base or "features"

def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def _coerce_numeric(val: str):
    if val is None:
        return 0.0
    val = val.strip()
    if val == "":
        return 0.0
    if "," in val and _is_float(val.replace(",", ".")):
        val = val.replace(",", ".")
    try:
        return float(val)
    except Exception:
        return 0.0

def _detect_column_types(rows, headers, id_col="patient_id"):
    numeric_cols, categorical_cols = [], []
    for h in headers:
        if h == id_col:
            continue
        col_vals = [r[h] for r in rows]
        non_empty = [v for v in col_vals if v is not None and v.strip() != ""]
        if non_empty and all(_is_float(v) or _is_float(v.replace(",", ".")) for v in non_empty):
            numeric_cols.append(h)
        else:
            categorical_cols.append(h)
    return numeric_cols, categorical_cols

def _build_one_hot_index(rows, categorical_cols):
    cat_map = {}
    for col in categorical_cols:
        cats = []
        seen = set()
        for r in rows:
            v = (r[col] or "").strip()
            if v == "":
                continue
            if v not in seen:
                seen.add(v)
                cats.append(v)
        cat_map[col] = {c: i for i, c in enumerate(sorted(cats))}
    return cat_map

def _rows_to_feature_matrix(rows, numeric_cols, categorical_cols, cat_map):
    N = len(rows)
    num_dims = len(numeric_cols)
    cat_dims = sum(len(cat_map[c]) for c in categorical_cols)
    D = num_dims + cat_dims
    X = np.zeros((N, D), dtype=np.float32)

    offsets = {}
    offset = num_dims
    for c in categorical_cols:
        offsets[c] = offset
        offset += len(cat_map[c])

    for i, r in enumerate(rows):
        for j, col in enumerate(numeric_cols):
            X[i, j] = _coerce_numeric(r[col])
        for c in categorical_cols:
            v = (r[c] or "").strip()
            if v == "":
                continue
            idx_map = cat_map[c]
            if v in idx_map:
                X[i, offsets[c] + idx_map[v]] = 1.0
    return X

def _features_from_mel_folder(npy_folder: str):
    """
    Load voiceXXX_stacked_mel.npy files and make a fixed-length vector per patient:
    concat(mean_over_time, std_over_time). Time axis is inferred as the larger dimension.
    Returns (patient_ids, features[npatients, 2*n_mels]).
    """
    files = sorted([f for f in os.listdir(npy_folder) if f.endswith(".npy")])
    if not files:
        raise ValueError(f"No .npy files found in {npy_folder}")

    patient_ids, feats = [], []
    for fname in files:
        # Expect "voice001_stacked_mel.npy" -> patient_id "voice001"
        pid = fname.split("_")[0]
        arr = np.load(os.path.join(npy_folder, fname), allow_pickle=False)

        if arr.ndim == 1:
            # 1D: treat as time series; produce [mean, std]
            mean_t = np.mean(arr, keepdims=True)
            std_t = np.std(arr, keepdims=True)
            feat = np.concatenate([mean_t, std_t], axis=0).astype(np.float32)
        elif arr.ndim == 2:
            # 2D mel: infer time axis as the larger dimension (typical: (n_mels, n_frames))
            time_axis = 1 if arr.shape[1] >= arr.shape[0] else 0
            mean_t = np.mean(arr, axis=time_axis)
            std_t = np.std(arr, axis=time_axis)
            feat = np.concatenate([mean_t, std_t], axis=0).astype(np.float32)
        else:
            # Higher dims: flatten all but last, treat last as time
            last_axis = arr.ndim - 1
            reshaped = arr.reshape(-1, arr.shape[last_axis])  # (freq_like, time)
            mean_t = np.mean(reshaped, axis=1)
            std_t = np.std(reshaped, axis=1)
            feat = np.concatenate([mean_t, std_t], axis=0).astype(np.float32)

        patient_ids.append(pid)
        feats.append(feat)

    # Align feature dimensionality across files (in case of rare mismatches)
    max_dim = max(f.shape[0] for f in feats)
    aligned = np.zeros((len(feats), max_dim), dtype=np.float32)
    for i, v in enumerate(feats):
        aligned[i, :v.shape[0]] = v  # right-pad with zeros if shorter

    return patient_ids, aligned

def _build_knn(embeddings: np.ndarray, k: int):
    num_patients = embeddings.shape[0]
    k_eff = min(k, max(1, num_patients - 1))
    sim_matrix = cosine_similarity(embeddings)
    edge_index, edge_attr = [], []
    for i in range(num_patients):
        sim_scores = sim_matrix[i]
        top_k = np.argsort(sim_scores)[::-1]
        top_k = [j for j in top_k if j != i][:k_eff]
        for j in top_k:
            edge_index.append([i, j])
            edge_attr.append(sim_scores[j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    node_features = torch.tensor(embeddings, dtype=torch.float32)
    return node_features, edge_index, edge_attr

def build_knn_patient_graph_from_csv(input_csv, output_folder, k=10, label_csv=None, label_column="label"):
    os.makedirs(output_folder, exist_ok=True)
    out_suffix = f"{_path_tag(input_csv)}_k{k}"

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if "patient_id" not in headers:
            raise ValueError("Expected a 'patient_id' column in the CSV.")
        rows = [row for row in reader]

    patient_ids = [r["patient_id"] for r in rows]
    numeric_cols, categorical_cols = _detect_column_types(rows, headers, id_col="patient_id")
    cat_map = _build_one_hot_index(rows, categorical_cols)
    embeddings = _rows_to_feature_matrix(rows, numeric_cols, categorical_cols, cat_map)

    labels = None
    if label_csv:
        patient_label_dict = {}
        with open(label_csv, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if "patient_id" not in r.fieldnames or label_column not in r.fieldnames:
                raise ValueError(f"label_csv must have columns: patient_id and {label_column}")
            for row in r:
                patient_label_dict[row["patient_id"]] = row[label_column]
        label_list = [patient_label_dict.get(pid, "unknown") for pid in patient_ids]
        unique_labels = sorted(set(label_list))
        label_to_int = {lab: i for i, lab in enumerate(unique_labels)}
        labels = torch.tensor([label_to_int[lab] for lab in label_list], dtype=torch.long)
        print(f"Labels loaded for {len(labels)} patients. Classes: {unique_labels}")

    x, edge_index, edge_attr = _build_knn(embeddings, k)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if labels is not None:
        data.y = labels

    graph_fname = f"patient_graph_{out_suffix}.pt"
    ids_fname = f"patient_ids_{out_suffix}.txt"

    torch.save(data, os.path.join(output_folder, graph_fname))
    with open(os.path.join(output_folder, ids_fname), "w", encoding="utf-8") as f:
        for pid in patient_ids:
            f.write(pid + "\n")

    print(f"Saved kNN graph with {len(patient_ids)} nodes and {edge_index.size(1)} edges to {os.path.join(output_folder, graph_fname)}")
    print(f"Saved patient id list to {os.path.join(output_folder, ids_fname)}")
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns (one-hot): {categorical_cols}")

def build_knn_patient_graph_from_mels(npy_folder, output_folder, k=10, label_csv=None, label_column="label"):
    os.makedirs(output_folder, exist_ok=True)
    out_suffix = f"{_path_tag(npy_folder)}_mel_k{k}"

    # Load mel features
    patient_ids, embeddings = _features_from_mel_folder(npy_folder)

    # Optional labels
    labels = None
    if label_csv:
        patient_label_dict = {}
        with open(label_csv, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if "patient_id" not in r.fieldnames or label_column not in r.fieldnames:
                raise ValueError(f"label_csv must have columns: patient_id and {label_column}")
            for row in r:
                patient_label_dict[row["patient_id"]] = row[label_column]
        label_list = [patient_label_dict.get(pid, "unknown") for pid in patient_ids]
        unique_labels = sorted(set(label_list))
        label_to_int = {lab: i for i, lab in enumerate(unique_labels)}
        labels = torch.tensor([label_to_int[lab] for lab in label_list], dtype=torch.long)
        print(f"Labels loaded for {len(labels)} patients. Classes: {unique_labels}")

    x, edge_index, edge_attr = _build_knn(embeddings, k)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if labels is not None:
        data.y = labels

    graph_fname = f"patient_graph_{out_suffix}.pt"
    ids_fname = f"patient_ids_{out_suffix}.txt"

    torch.save(data, os.path.join(output_folder, graph_fname))
    with open(os.path.join(output_folder, ids_fname), "w", encoding="utf-8") as f:
        for pid in patient_ids:
            f.write(pid + "\n")

    print(f"Saved kNN graph with {len(patient_ids)} nodes and {edge_index.size(1)} edges to {os.path.join(output_folder, graph_fname)}")
    print(f"Saved patient id list to {os.path.join(output_folder, ids_fname)}")
    print(f"Feature dimension per node: {x.shape[1]} (concatenated mean+std across time)")

def main():
    parser = argparse.ArgumentParser(description="Build patient kNN graph from CSV clinical features or Mel spectrogram .npy files.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_csv", type=str, help="Path to clinical_features.csv")
    group.add_argument("--npy_folder", type=str, help="Folder containing voiceXXX_stacked_mel.npy files")

    parser.add_argument("--output_folder", type=str, required=True, help="Where to save patient graph")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors")
    parser.add_argument("--label_csv", type=str, default=None, help="Optional CSV with columns: patient_id,label")
    parser.add_argument("--label_column", type=str, default="label", help="Label column name in label_csv")
    args = parser.parse_args()

    if args.input_csv:
        build_knn_patient_graph_from_csv(
            args.input_csv, args.output_folder, k=args.k,
            label_csv=args.label_csv, label_column=args.label_column
        )
    else:
        build_knn_patient_graph_from_mels(
            args.npy_folder, args.output_folder, k=args.k,
            label_csv=args.label_csv, label_column=args.label_column
        )

if __name__ == "__main__":
    main()
