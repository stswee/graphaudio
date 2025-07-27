import argparse
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from tqdm import tqdm
import pandas as pd
import os
import numpy as np

from models.gcn import TextGCN, MelGCN, MultiModalGCN
from models.gat import TextGAT, MelGAT, MultiModalGAT
from models.graphsage import TextGraphSAGE, MelGraphSAGE, MultiModalGraphSAGE

def get_model(architecture, modality, input_dim_text, input_dim_mel):
    if architecture == 'GCN':
        if modality == 'text':
            return TextGCN(input_dim_text=input_dim_text)
        elif modality == 'mel':
            return MelGCN(input_dim_mel=input_dim_mel)
        elif modality == 'both':
            return MultiModalGCN(input_dim_text=input_dim_text, input_dim_mel=input_dim_mel)
    elif architecture == 'GAT':
        if modality == 'text':
            return TextGAT(input_dim_text=input_dim_text)
        elif modality == 'mel':
            return MelGAT(input_dim_mel=input_dim_mel)
        elif modality == 'both':
            return MultiModalGAT(input_dim_text=input_dim_text, input_dim_mel=input_dim_mel)
    elif architecture == 'GraphSAGE':
        if modality == 'text':
            return TextGraphSAGE(input_dim_text=input_dim_text)
        elif modality == 'mel':
            return MelGraphSAGE(input_dim_mel=input_dim_mel)
        elif modality == 'both':
            return MultiModalGraphSAGE(input_dim_text=input_dim_text, input_dim_mel=input_dim_mel)
    else:
        raise ValueError("Invalid architecture or modality")

def load_data_from_patient_files(label_csv, text_dir, mel_dir, classification):
    df = pd.read_csv(label_csv)
    df = df.sort_values(by='patient_id')  # Ensures consistent order
    patient_ids = df['patient_id'].tolist()

    if classification == 'binary':
        # Binary: healthy=0, all others=1
        df['label'] = df['label'].apply(lambda x: 0 if str(x).lower() == 'healthy' else 1)
        labels = torch.tensor(df['label'].values.astype(np.int64), dtype=torch.long)
        print("Label distribution (0=healthy, 1=disease):", torch.bincount(labels).tolist())
    elif classification == 'multiclass':
        # Convert string labels to integer categories
        df['label'] = df['label'].astype('category').cat.codes
        labels = torch.tensor(df['label'].values.astype(np.int64), dtype=torch.long)
        print("Multiclass label distribution:", torch.bincount(labels).tolist())
    else:
        raise ValueError("classification argument must be 'binary' or 'multiclass'")

    text_embeddings = []
    mel_raw = []

    print("Loading patient data...")
    for pid in tqdm(patient_ids, desc="Loading embeddings"):
        text_path = os.path.join(text_dir, f"{pid}-info.pt")
        mel_path = os.path.join(mel_dir, f"{pid}_stacked_mel.npy")

        if not os.path.exists(text_path) or not os.path.exists(mel_path):
            raise FileNotFoundError(f"Missing data for patient {pid}")

        text = torch.load(text_path)
        if len(text.shape) > 1:
            text = text.view(-1)
        text_embeddings.append(text)

        mel = np.load(mel_path)
        mel_tensor = torch.tensor(mel, dtype=torch.float32).flatten()
        mel_raw.append(mel_tensor)

    max_mel_len = max(m.shape[0] for m in mel_raw)
    mel_embeddings = [F.pad(m, (0, max_mel_len - m.shape[0])) for m in mel_raw]

    text_tensor = torch.stack(text_embeddings)
    mel_tensor = torch.stack(mel_embeddings)

    return text_tensor, mel_tensor, labels

def evaluate(model, data, val_idx, labels, modality, device, classification):
    model.eval()
    with torch.no_grad():
        if modality == 'both':
            out = model(data.text.to(device), data.mel.to(device), data.edge_index.to(device))
        elif modality == 'text':
            out = model(data.text.to(device), data.edge_index.to(device))
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
                auc = roc_auc_score(true.cpu(), prob_positive.cpu())
            except Exception:
                auc = float('nan')
        else:  # multiclass
            prob_positive = probs
            try:
                auc = roc_auc_score(true.cpu(), prob_positive.cpu(), multi_class='ovr')
            except Exception:
                auc = float('nan')

        acc = accuracy_score(true.cpu(), preds.cpu())
        precision, recall, f1, _ = precision_recall_fscore_support(true.cpu(), preds.cpu(), average='weighted')

        return acc, precision, recall, f1, auc, preds.cpu(), true.cpu()

def main(args):
    graph_file = os.path.join(args.graph_dir, 'patient_graph.pt')
    if not os.path.exists(graph_file):
        raise FileNotFoundError(f"Graph file not found in {graph_file}")

    graph_data = torch.load(graph_file, weights_only=False)

    text_embeddings, mel_embeddings, labels = load_data_from_patient_files(
        args.label_csv, args.text_dir, args.mel_dir, args.classification
    )

    graph_data.text = text_embeddings
    graph_data.mel = mel_embeddings
    graph_data.y = labels

    input_dim_text = text_embeddings.shape[1]
    input_dim_mel = mel_embeddings.shape[1]

    kf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
    metrics = {'acc': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

    for fold, (train_idx, val_idx) in enumerate(kf.split(graph_data.x, labels.cpu().numpy())):
        print(f'\nFold {fold + 1}/{args.k}')
        model = get_model(args.arch, args.modality, input_dim_text, input_dim_mel).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in tqdm(range(args.epochs), desc=f"Training Fold {fold+1}", leave=False):
            model.train()
            optimizer.zero_grad()

            if args.modality == 'both':
                out = model(graph_data.text.to(args.device), graph_data.mel.to(args.device), graph_data.edge_index.to(args.device))
            elif args.modality == 'text':
                out = model(graph_data.text.to(args.device), graph_data.edge_index.to(args.device))
            elif args.modality == 'mel':
                out = model(graph_data.mel.to(args.device), graph_data.edge_index.to(args.device))
            else:
                raise ValueError("Invalid modality")

            loss = F.cross_entropy(out[train_idx].to(args.device), labels[train_idx].to(args.device))
            loss.backward()
            optimizer.step()

        acc, precision, recall, f1, auc, preds, true = evaluate(model, graph_data, val_idx, labels, args.modality, args.device, args.classification)
        print(f"[Fold {fold+1}] Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        cm = confusion_matrix(true, preds)
        print(f"Confusion Matrix for Fold {fold+1}:\n{cm}")

        metrics['acc'].append(acc)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        metrics['auc'].append(auc)

    print("\n=== Final Results ===")
    for metric_name, scores in metrics.items():
        avg = sum(scores) / len(scores)
        print(f"{metric_name.capitalize()}: {avg:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, choices=['GCN', 'GAT', 'GraphSAGE'], required=True)
    parser.add_argument('--modality', type=str, choices=['text', 'mel', 'both'], required=True)
    parser.add_argument('--classification', type=str, choices=['binary', 'multiclass'], default='binary', help="Choose classification type")
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--graph_dir', type=str, required=True, help="Directory containing patient_graph.pt")
    parser.add_argument('--text_dir', type=str, required=True, help="Directory with voiceXXX-info.pt")
    parser.add_argument('--mel_dir', type=str, required=True, help="Directory with voiceXXX_stacked_mel.npy")
    parser.add_argument('--label_csv', type=str, required=True, help="CSV with columns patient_id,label")

    args = parser.parse_args()
    main(args)
