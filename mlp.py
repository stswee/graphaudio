import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from tqdm import tqdm
import pandas as pd
import os
import numpy as np

# Simple MLP baseline
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=4):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def load_data_from_patient_files(label_csv, text_dir, mel_dir):
    df = pd.read_csv(label_csv)
    df = df.sort_values(by='patient_id')
    patient_ids = df['patient_id'].tolist()

    df['label'] = df['label'].astype('category').cat.codes
    labels = torch.tensor(df['label'].values.astype(np.int64), dtype=torch.long)

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

def evaluate(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        logits = model(X_val)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        acc = accuracy_score(y_val.cpu(), preds.cpu())
        precision, recall, f1, _ = precision_recall_fscore_support(y_val.cpu(), preds.cpu(), average='weighted', zero_division=0)

        try:
            auc = roc_auc_score(y_val.cpu(), probs.cpu(), multi_class='ovr')
        except:
            auc = float('nan')

        cm = confusion_matrix(y_val.cpu(), preds.cpu())
        return acc, precision, recall, f1, auc, cm

def main(args):
    text, mel, labels = load_data_from_patient_files(args.label_csv, args.text_dir, args.mel_dir)

    if args.modality == 'text':
        X = text
    elif args.modality == 'mel':
        X = mel
    elif args.modality == 'both':
        X = torch.cat([text, mel], dim=1)
    else:
        raise ValueError("Invalid modality")

    kf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
    metrics = {'acc': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, labels)):
        print(f"\nFold {fold + 1}/{args.k}")
        model = MLP(input_dim=X.shape[1]).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        X_train, y_train = X[train_idx].to(args.device), labels[train_idx].to(args.device)
        X_val, y_val = X[val_idx].to(args.device), labels[val_idx].to(args.device)

        for epoch in tqdm(range(args.epochs), desc=f"Training Fold {fold+1}", leave=False):
            model.train()
            optimizer.zero_grad()
            logits = model(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

        acc, precision, recall, f1, auc, cm = evaluate(model, X_val, y_val)
        print(f"[Fold {fold+1}] Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        print(f"Confusion Matrix:\n{cm}")

        metrics['acc'].append(acc)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        metrics['auc'].append(auc)

    print("\n=== Final MLP Results ===")
    for metric_name, scores in metrics.items():
        avg = sum(scores) / len(scores)
        print(f"{metric_name.capitalize()}: {avg:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, choices=['text', 'mel', 'both'], required=True)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--text_dir', type=str, required=True)
    parser.add_argument('--mel_dir', type=str, required=True)
    parser.add_argument('--label_csv', type=str, required=True)

    args = parser.parse_args()
    main(args)
