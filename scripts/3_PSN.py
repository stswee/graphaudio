import os
import argparse
import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import csv

def build_knn_patient_graph(input_folder, output_folder, k=5, label_csv=None):
    os.makedirs(output_folder, exist_ok=True)

    embeddings = []
    patient_ids = []

    print("Loading patient embeddings...")
    for fname in tqdm(sorted(os.listdir(input_folder))):
        if fname.endswith("-info.pt"):
            pid = fname.split("-")[0]
            path = os.path.join(input_folder, fname)
            emb = torch.load(path)  # Tensor: (seq_len, embed_dim)
    
            if isinstance(emb, dict):
                raise ValueError(f"Expected a tensor in {fname}, but got a dict.")
    
            emb_np = emb.mean(dim=0).numpy()  # mean-pool across tokens → (embedding_dim,)
            embeddings.append(emb_np)
            patient_ids.append(pid)

    embeddings = np.array(embeddings)
    num_patients = embeddings.shape[0]

    # Load labels if provided
    labels = None
    if label_csv:
        print(f"Loading labels from {label_csv} ...")
        patient_label_dict = {}
        with open(label_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_label_dict[row['patient_id']] = row['label']

        # Extract labels in the same order as patient_ids
        label_list = []
        for pid in patient_ids:
            label = patient_label_dict.get(pid, "unknown")
            label_list.append(label)

        # Convert string labels to integer classes
        unique_labels = sorted(set(label_list))
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        labels = [label_to_int[label] for label in label_list]
        labels = torch.tensor(labels, dtype=torch.long)

        print(f"Labels loaded for {len(labels)} patients. Classes: {unique_labels}")

    print(f"Computing cosine similarity and building kNN graph with k={k}...")
    sim_matrix = cosine_similarity(embeddings)

    edge_index = []
    edge_attr = []

    for i in range(num_patients):
        sim_scores = sim_matrix[i]
        top_k = np.argsort(sim_scores)[::-1][1:k+1]  # Exclude self, get top-k

        for j in top_k:
            edge_index.append([i, j])
            edge_attr.append(sim_scores[j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    node_features = torch.tensor(embeddings, dtype=torch.float)

    if labels is not None:
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=labels)
    else:
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    torch.save(data, os.path.join(output_folder, "patient_graph.pt"))
    with open(os.path.join(output_folder, "patient_ids.txt"), "w") as f:
        for pid in patient_ids:
            f.write(pid + "\n")

    print(f"\nSaved kNN graph with {num_patients} nodes and {edge_index.size(1)} edges to {output_folder}/patient_graph.pt")

def main():
    parser = argparse.ArgumentParser(description="Build patient kNN graph from .pt text embeddings.")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder with .pt embeddings")
    parser.add_argument("--output_folder", type=str, required=True, help="Where to save patient graph")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors")
    parser.add_argument("--label_csv", type=str, default=None, help="CSV file with patient labels (optional)")
    args = parser.parse_args()

    build_knn_patient_graph(args.input_folder, args.output_folder, k=args.k, label_csv=args.label_csv)

if __name__ == "__main__":
    main()
