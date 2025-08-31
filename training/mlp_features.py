import argparse
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, roc_curve, auc as sk_auc
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from models.gcn import ClinicalGCN, MelGCN, MultiModalClinicalGCN
from models.gat import ClinicalGAT, MelGAT, MultiModalClinicalGAT
from models.graphsage import ClinicalGraphSAGE, MelGraphSAGE, MultiModalClinicalGraphSAGE
import random

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------- MLP baseline --------------------
class MLP(nn.Module):
    """
    MLP with exactly n_hidden_layers fully-connected layers (not counting output).
    """
    def __init__(self, in_dim: int, out_dim: int, n_hidden_layers: int, hidden_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.is_mlp = True  # flag to branch train/eval paths
        layers = []
        prev = in_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = hidden_dim
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(prev, out_dim)

    def forward(self, x):
        return self.head(self.backbone(x))

def _mlp_layers_for_modality(modality: str) -> int:
    # Match your table: Clinical=2, Mel=3, Clinical+Mel=4
    if modality == 'clinical':
        return 2
    if modality == 'mel':
        return 3
    if modality == 'both':
        return 4
    raise ValueError("Invalid modality for MLP")

# -------------------- Model picker --------------------
def get_model(architecture, modality, input_dim_clinical, input_dim_mel, num_classes,
              mlp_hidden_dim=512, mlp_dropout=0.0):
    if architecture == 'GCN':
        if modality == 'clinical':
            return ClinicalGCN(input_dim_clinical=input_dim_clinical)
        elif modality == 'mel':
            return MelGCN(input_dim_mel=input_dim_mel)
        elif modality == 'both':
            return MultiModalClinicalGCN(input_dim_clinical=input_dim_clinical, input_dim_mel=input_dim_mel)

    elif architecture == 'GAT':
        if modality == 'clinical':
            return ClinicalGAT(input_dim_clinical=input_dim_clinical)
        elif modality == 'mel':
            return MelGAT(input_dim_mel=input_dim_mel)
        elif modality == 'both':
            return MultiModalClinicalGAT(input_dim_clinical=input_dim_clinical, input_dim_mel=input_dim_mel)

    elif architecture == 'GraphSAGE':
        if modality == 'clinical':
            return ClinicalGraphSAGE(input_dim_clinical=input_dim_clinical)
        elif modality == 'mel':
            return MelGraphSAGE(input_dim_mel=input_dim_mel)
        elif modality == 'both':
            return MultiModalClinicalGraphSAGE(input_dim_clinical=input_dim_clinical, input_dim_mel=input_dim_mel)

    elif architecture == 'MLP':
        if modality == 'clinical':
            in_dim = input_dim_clinical
        elif modality == 'mel':
            in_dim = input_dim_mel
        elif modality == 'both':
            in_dim = input_dim_clinical + input_dim_mel
        else:
            raise ValueError("Invalid modality")
        n_hidden_layers = _mlp_layers_for_modality(modality)
        return MLP(in_dim=in_dim, out_dim=num_classes, n_hidden_layers=n_hidden_layers,
                   hidden_dim=mlp_hidden_dim, dropout=mlp_dropout)

    else:
        raise ValueError("Invalid architecture or modality")

# -------------------- Helpers --------------------
def _slugify(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)

def _run_slug(args) -> str:
    graph_tag = Path(args.graph_file).stem
    clin_tag  = Path(args.clinical_csv).name
    return _slugify(f"{args.arch}__{args.modality}__{args.classification}__{graph_tag}__{clin_tag}")

def _interpolate_mean_std(fprs_tprs, num_points=200):
    if not fprs_tprs:
        return None
    mean_fpr = np.linspace(0.0, 1.0, num_points)
    tprs, aucs = [], []
    for fpr, tpr in fprs_tprs:
        fpr = np.asarray(fpr, dtype=float)
        tpr = np.asarray(tpr, dtype=float)
        if fpr.ndim != 1 or tpr.ndim != 1 or len(fpr) < 2:
            continue
        tpr_i = np.interp(mean_fpr, fpr, tpr)
        tpr_i[0] = 0.0
        tprs.append(tpr_i)
        aucs.append(sk_auc(fpr, tpr))
    if not tprs:
        return None
    tprs = np.vstack(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr  = tprs.std(axis=0, ddof=1) if tprs.shape[0] > 1 else np.zeros_like(mean_tpr)
    mean_tpr[-1] = 1.0
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    std_auc  = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
    return mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc

def _aggregate_multiclass_ovr(fold_true_list, fold_proba_list, n_classes, num_points=200):
    per_class_curves = {c: [] for c in range(n_classes)}
    micro_curves = []
    for y_true, y_proba in zip(fold_true_list, fold_proba_list):
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        for c in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_proba[:, c])
            per_class_curves[c].append((fpr, tpr))
        fpr_mi, tpr_mi, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
        micro_curves.append((fpr_mi, tpr_mi))
    per_class_stats = {}
    for c in range(n_classes):
        agg = _interpolate_mean_std(per_class_curves[c], num_points=num_points)
        if agg is None:
            continue
        mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc = agg
        per_class_stats[c] = dict(mean_fpr=mean_fpr, mean_tpr=mean_tpr, std_tpr=std_tpr,
                                  mean_auc=mean_auc, std_auc=std_auc)
    valid = sorted(per_class_stats.keys())
    if valid:
        mean_fpr = per_class_stats[valid[0]]["mean_fpr"]
        stack = np.vstack([per_class_stats[c]["mean_tpr"] for c in valid])
        macro_mean_tpr = stack.mean(axis=0)
        macro_std_tpr  = stack.std(axis=0, ddof=1) if len(valid) > 1 else np.zeros_like(macro_mean_tpr)
        macro_mean_auc = float(np.mean([per_class_stats[c]["mean_auc"] for c in valid]))
        macro_std_auc  = float(np.std([per_class_stats[c]["mean_auc"] for c in valid], ddof=1)) if len(valid) > 1 else 0.0
    else:
        mean_fpr = np.linspace(0, 1, num_points)
        macro_mean_tpr = np.zeros_like(mean_fpr)
        macro_std_tpr  = np.zeros_like(mean_fpr)
        macro_mean_auc = float("nan")
        macro_std_auc  = 0.0
    m_agg = _interpolate_mean_std(micro_curves, num_points=num_points)
    if m_agg is None:
        micro_mean_fpr = mean_fpr
        micro_mean_tpr = np.zeros_like(mean_fpr)
        micro_std_tpr  = np.zeros_like(mean_fpr)
        micro_mean_auc = float("nan")
        micro_std_auc  = 0.0
    else:
        micro_mean_fpr, micro_mean_tpr, micro_std_tpr, micro_mean_auc, micro_std_auc = m_agg
    return {
        "per_class": per_class_stats,
        "macro": {
            "mean_fpr": mean_fpr,
            "mean_tpr": macro_mean_tpr,
            "std_tpr":  macro_std_tpr,
            "mean_auc": macro_mean_auc,
            "std_auc":  macro_std_auc,
        },
        "micro": {
            "mean_fpr": micro_mean_fpr,
            "mean_tpr": micro_mean_tpr,
            "std_tpr":  micro_std_tpr,
            "mean_auc": micro_mean_auc,
            "std_auc":  micro_std_auc,
        }
    }

# -------------------- Data loading & eval --------------------
def _infer_numeric_cols(df: pd.DataFrame):
    numeric_cols, categorical_cols = [], []
    for col in df.columns:
        series = df[col]
        non_empty = series[series.notna() & (series.astype(str).str.strip() != "")]
        def _is_floatable(x):
            s = str(x).strip().replace(",", ".")
            try:
                float(s)
                return True
            except Exception:
                return False
        if len(non_empty) > 0 and non_empty.map(_is_floatable).all():
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols

def load_data_from_patient_csv(label_csv, clinical_csv, mel_dir, classification):
    # Labels
    df_labels = pd.read_csv(label_csv).sort_values(by='patient_id')
    patient_ids = df_labels['patient_id'].tolist()

    if classification == 'binary':
        df_labels['label'] = df_labels['label'].apply(lambda x: 0 if str(x).lower() == 'healthy' else 1)
        labels = torch.tensor(df_labels['label'].values.astype(np.int64), dtype=torch.long)
        print("Label distribution (0=healthy, 1=disease):", torch.bincount(labels).tolist())
    elif classification == 'multiclass':
        df_labels['label'] = df_labels['label'].astype('category').cat.codes
        labels = torch.tensor(df_labels['label'].values.astype(np.int64), dtype=torch.long)
        print("Multiclass label distribution:", torch.bincount(labels).tolist())
    else:
        raise ValueError("classification argument must be 'binary' or 'multiclass'")

    # Clinical features
    df_feat = pd.read_csv(clinical_csv)
    if 'patient_id' not in df_feat.columns:
        raise ValueError("clinical_csv must contain a 'patient_id' column")
    df_feat = df_feat.set_index('patient_id')

    missing = [pid for pid in patient_ids if pid not in df_feat.index]
    if missing:
        raise ValueError(f"Missing patient_ids in clinical_csv: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    df_feat = df_feat.loc[patient_ids]

    numeric_cols, categorical_cols = _infer_numeric_cols(df_feat)

    df_num = df_feat[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=df_feat.index)
    for c in df_num.columns:
        df_num[c] = df_num[c].astype(str).str.replace(",", ".", regex=False)
        df_num[c] = pd.to_numeric(df_num[c], errors='coerce').fillna(0.0)

    df_cat = df_feat[categorical_cols].copy() if categorical_cols else pd.DataFrame(index=df_feat.index)
    if not df_cat.empty:
        df_cat = df_cat.applymap(lambda x: ("" if pd.isna(x) else str(x)).strip())
        df_cat = df_cat.replace({"": np.nan})
        df_cat = pd.get_dummies(df_cat, dummy_na=False)
        df_cat = df_cat.astype(np.float32)

    if df_num.empty and df_cat.empty:
        raise ValueError("No features found after processing clinical_csv.")
    X = pd.concat([df_num, df_cat], axis=1).astype(np.float32).fillna(0.0)
    clinical_tensor = torch.tensor(X.values, dtype=torch.float32)

    # Mel features
    mel_raw = []
    print("Loading mel data...")
    for pid in tqdm(patient_ids, desc="Loading mels"):
        mel_path = os.path.join(mel_dir, f"{pid}_stacked_mel.npy")
        if not os.path.exists(mel_path):
            raise FileNotFoundError(f"Missing mel file for patient {pid}: {mel_path}")
        mel = np.load(mel_path)
        mel_tensor = torch.tensor(mel, dtype=torch.float32).flatten()
        mel_raw.append(mel_tensor)

    max_mel_len = max(m.shape[0] for m in mel_raw)
    mel_embeddings = [F.pad(m, (0, max_mel_len - m.shape[0])) for m in mel_raw]
    mel_tensor = torch.stack(mel_embeddings)

    return clinical_tensor, mel_tensor, labels

def evaluate(model, data, val_idx, labels, modality, device, classification):
    model.eval()
    with torch.no_grad():
        if getattr(model, "is_mlp", False):
            if modality == 'both':
                x = torch.cat([data.clinical, data.mel], dim=1).to(device)
            elif modality == 'clinical':
                x = data.clinical.to(device)
            elif modality == 'mel':
                x = data.mel.to(device)
            else:
                raise ValueError("Invalid modality")
            out = model(x)
        else:
            if modality == 'both':
                out = model(data.clinical.to(device), data.mel.to(device), data.edge_index.to(device))
            elif modality == 'clinical':
                out = model(data.clinical.to(device), data.edge_index.to(device))
            elif modality == 'mel':
                out = model(data.mel.to(device), data.edge_index.to(device))
            else:
                raise ValueError("Invalid modality")

        preds = out[val_idx].argmax(dim=1)
        true = labels[val_idx]

        probs = F.softmax(out[val_idx], dim=1)
        if classification == 'binary':
            prob_positive = probs[:, 1]
            try:
                auc_val = roc_auc_score(true.cpu(), prob_positive.cpu())
            except Exception:
                auc_val = float('nan')
            proba_for_roc = prob_positive.cpu()  # 1D
        else:
            try:
                auc_val = roc_auc_score(true.cpu(), probs.cpu(), multi_class='ovr')
            except Exception:
                auc_val = float('nan')
            proba_for_roc = probs.cpu()  # 2D (N,C)

        acc = accuracy_score(true.cpu(), preds.cpu())
        precision, recall, f1, _ = precision_recall_fscore_support(true.cpu(), preds.cpu(), average='weighted')

        return acc, precision, recall, f1, auc_val, preds.cpu(), true.cpu(), proba_for_roc

# -------------------- Main --------------------
def main(args):
    # Graph
    if not os.path.exists(args.graph_file):
        raise FileNotFoundError(f"Graph file not found: {args.graph_file}")
    graph_data = torch.load(args.graph_file, weights_only=False)

    # Output files
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    slug = _run_slug(args)
    summary_path = results_dir / f"summary_{slug}.tsv"
    roc_path     = results_dir / f"roc_{slug}.npz"
    rawjson_path = results_dir / f"raw_{slug}.json"

    # Data
    clinical_embeddings, mel_embeddings, labels = load_data_from_patient_csv(
        args.label_csv, args.clinical_csv, args.mel_dir, args.classification
    )
    graph_data.clinical = clinical_embeddings
    graph_data.mel = mel_embeddings
    graph_data.y = labels

    input_dim_clinical = clinical_embeddings.shape[1]
    input_dim_mel = mel_embeddings.shape[1]
    num_classes = int(labels.max().item() + 1)

    # KFold split (do NOT rely on graph_data.x)
    N = labels.shape[0]
    kf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
    split_iter = kf.split(np.zeros(N), labels.cpu().numpy())

    metrics = {'acc': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

    # ROC collectors
    fold_rocs_bin = []   # list of (fpr, tpr) for binary
    mc_fold_true = []    # list of arrays (N_fold,)
    mc_fold_proba = []   # list of arrays (N_fold, C)
    n_classes_seen = None

    for fold, (train_idx, val_idx) in enumerate(split_iter):
        print(f'\nFold {fold + 1}/{args.k}')
        model = get_model(
            args.arch, args.modality, input_dim_clinical, input_dim_mel, num_classes,
            mlp_hidden_dim=args.mlp_hidden_dim, mlp_dropout=args.mlp_dropout
        ).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in tqdm(range(args.epochs), desc=f"Training Fold {fold+1}", leave=False):
            model.train()
            optimizer.zero_grad()

            if getattr(model, "is_mlp", False):
                if args.modality == 'both':
                    x = torch.cat([graph_data.clinical, graph_data.mel], dim=1).to(args.device)
                elif args.modality == 'clinical':
                    x = graph_data.clinical.to(args.device)
                elif args.modality == 'mel':
                    x = graph_data.mel.to(args.device)
                else:
                    raise ValueError("Invalid modality")
                out = model(x)
            else:
                if args.modality == 'both':
                    out = model(
                        graph_data.clinical.to(args.device),
                        graph_data.mel.to(args.device),
                        graph_data.edge_index.to(args.device)
                    )
                elif args.modality == 'clinical':
                    out = model(graph_data.clinical.to(args.device), graph_data.edge_index.to(args.device))
                elif args.modality == 'mel':
                    out = model(graph_data.mel.to(args.device), graph_data.edge_index.to(args.device))
                else:
                    raise ValueError("Invalid modality")

            loss = F.cross_entropy(out[train_idx].to(args.device), labels[train_idx].to(args.device))
            loss.backward()
            optimizer.step()

        acc, precision, recall, f1, auc_val, preds, true, proba = evaluate(
            model, graph_data, val_idx, labels, args.modality, args.device, args.classification
        )
        print(f"[Fold {fold+1}] Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc_val:.4f}")

        cm = confusion_matrix(true, preds)
        print(f"Confusion Matrix for Fold {fold+1}:\n{cm}")

        for m, v in zip(['acc', 'precision', 'recall', 'f1', 'auc'], [acc, precision, recall, f1, auc_val]):
            metrics[m].append(v)

        # Collect ROC info
        if args.classification == 'binary':
            y_true_np = true.numpy()
            y_score_np = proba.numpy()  # 1D pos-class prob
            fpr, tpr, _ = roc_curve(y_true_np, y_score_np)
            fold_rocs_bin.append((fpr, tpr))
        else:
            y_true_np = true.numpy()
            y_proba_np = proba.numpy()  # 2D (N,C)
            mc_fold_true.append(y_true_np)
            mc_fold_proba.append(y_proba_np)
            if n_classes_seen is None:
                n_classes_seen = y_proba_np.shape[1]

    # ---- Summary metrics (mean ± std) ----
    means = {k: float(np.mean(v)) if len(v) else float("nan") for k, v in metrics.items()}
    stds  = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in metrics.items()}

    print("\n=== Final Results ===")
    for k in ['acc','precision','recall','f1','auc']:
        print(f"{k.capitalize()}: {means[k]:.4f} ± {stds[k]:.4f}")

    # Save summary TSV
    df = pd.DataFrame([{
        "arch": args.arch,
        "modality": args.modality,
        "classification": args.classification,
        "graph_file": args.graph_file,
        "clinical_csv": args.clinical_csv,
        "mel_dir": args.mel_dir,
        **{f"{k}_mean": means[k] for k in metrics},
        **{f"{k}_std": stds[k] for k in metrics},
    }])
    df.to_csv(summary_path, sep="\t", index=False)
    print(f"[OK] Wrote summary → {summary_path}")

    # Raw JSON (fold metrics only)
    raw_payload = {
        "fold_metrics": [
            {"acc": float(a), "precision": float(p), "recall": float(r),
             "f1": float(f), "auc": float(u)}
            for a,p,r,f,u in zip(metrics['acc'], metrics['precision'],
                                 metrics['recall'], metrics['f1'], metrics['auc'])
        ],
        "meta": {
            "arch": args.arch,
            "modality": args.modality,
            "classification": args.classification,
            "graph_file": args.graph_file,
            "clinical_csv": args.clinical_csv,
            "mel_dir": args.mel_dir,
        }
    }
    with open(rawjson_path, "w", encoding="utf-8") as f:
        json.dump(raw_payload, f)
    print(f"[OK] Wrote raw fold data → {rawjson_path}")

    # ---- Save ROC mean ± std ----
    if args.classification == 'binary' and fold_rocs_bin:
        agg = _interpolate_mean_std(fold_rocs_bin, num_points=200)
        if agg is not None:
            mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc = agg
            np.savez_compressed(
                roc_path,
                kind="binary",
                mean_fpr=mean_fpr, mean_tpr=mean_tpr, std_tpr=std_tpr,
                mean_auc=float(mean_auc), std_auc=float(std_auc)
            )
            print(f"[OK] Wrote binary mean±std ROC → {roc_path}")
        else:
            print("[WARN] Not enough ROC points to aggregate; skipping ROC save.")

    elif args.classification == 'multiclass' and mc_fold_true and mc_fold_proba and n_classes_seen:
        mc = _aggregate_multiclass_ovr(mc_fold_true, mc_fold_proba, n_classes_seen, num_points=200)
        classes = sorted(mc["per_class"].keys())
        if classes:
            mean_fpr = mc["per_class"][classes[0]]["mean_fpr"]
            per_class_mean_tpr = np.vstack([mc["per_class"][c]["mean_tpr"] for c in classes])
            per_class_std_tpr  = np.vstack([mc["per_class"][c]["std_tpr"]  for c in classes])
            per_class_mean_auc = np.array([mc["per_class"][c]["mean_auc"] for c in classes], dtype=float)
            per_class_std_auc  = np.array([mc["per_class"][c]["std_auc"] for c in classes], dtype=float)
        else:
            mean_fpr = np.linspace(0, 1, 200)
            per_class_mean_tpr = np.zeros((0, len(mean_fpr)))
            per_class_std_tpr  = np.zeros((0, len(mean_fpr)))
            per_class_mean_auc = np.zeros((0,))
            per_class_std_auc  = np.zeros((0,))

        np.savez_compressed(
            roc_path,
            kind="multiclass_ovr",
            classes=np.array(classes, dtype=int),
            mean_fpr=mean_fpr,
            per_class_mean_tpr=per_class_mean_tpr,
            per_class_std_tpr=per_class_std_tpr,
            per_class_mean_auc=per_class_mean_auc,
            per_class_std_auc=per_class_std_auc,
            macro_mean_fpr=mc["macro"]["mean_fpr"],
            macro_mean_tpr=mc["macro"]["mean_tpr"],
            macro_std_tpr=mc["macro"]["std_tpr"],
            macro_mean_auc=float(mc["macro"]["mean_auc"]),
            macro_std_auc=float(mc["macro"]["std_auc"]),
            micro_mean_fpr=mc["micro"]["mean_fpr"],
            micro_mean_tpr=mc["micro"]["mean_tpr"],
            micro_std_tpr=mc["micro"]["std_tpr"],
            micro_mean_auc=float(mc["micro"]["mean_auc"]),
            micro_std_auc=float(mc["micro"]["std_auc"]),
        )
        print(f"[OK] Wrote multiclass OvR mean±std ROC → {roc_path}")
    else:
        print("[INFO] No ROC data available to save.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, choices=['GCN', 'GAT', 'GraphSAGE', 'MLP'], required=True)
    parser.add_argument('--modality', type=str, choices=['clinical', 'mel', 'both'], required=True)
    parser.add_argument('--classification', type=str, choices=['binary', 'multiclass'], default='binary')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--graph_file', type=str, required=True, help="Path to patient_graph.pt")
    parser.add_argument('--clinical_csv', type=str, required=True, help="Path to clinical_features.csv")
    parser.add_argument('--mel_dir', type=str, required=True, help="Directory with *_stacked_mel.npy files")
    parser.add_argument('--label_csv', type=str, required=True, help="CSV with columns patient_id,label")

    # Output dir
    parser.add_argument('--results_dir', type=str, default='./runs', help="Directory to store summary/ROC outputs")

    # MLP hyperparams (depth is auto-set by modality for fair comparison-by-layers)
    parser.add_argument('--mlp_hidden_dim', type=int, default=512)
    parser.add_argument('--mlp_dropout', type=float, default=0.0)

    args = parser.parse_args()
    set_seed(42)
    main(args)
