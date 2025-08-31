#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, roc_curve, auc as sk_auc
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from models.gcn import TextGCN, MelGCN, MultiModalGCN
from models.gat import TextGAT, MelGAT, MultiModalGAT
from models.graphsage import TextGraphSAGE, MelGraphSAGE, MultiModalGraphSAGE
import random

# NEW: plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------- Model picker --------------------
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
    raise ValueError("Invalid architecture or modality")

# -------------------- Helpers --------------------
def _slugify(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)

def _run_slug(args) -> str:
    graph_tag = Path(args.graph_file).stem
    text_tag = Path(args.text_dir).name
    return _slugify(f"{args.arch}__{args.modality}__{args.classification}__{graph_tag}__{text_tag}")

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
        per_class_stats[c] = dict(
            mean_fpr=mean_fpr, mean_tpr=mean_tpr, std_tpr=std_tpr,
            mean_auc=mean_auc, std_auc=std_auc
        )
    valid_classes = sorted(per_class_stats.keys())
    if valid_classes:
        mean_fpr = per_class_stats[valid_classes[0]]["mean_fpr"]
        stack_mean_tpr = np.vstack([per_class_stats[c]["mean_tpr"] for c in valid_classes])
        macro_mean_tpr = stack_mean_tpr.mean(axis=0)
        macro_std_tpr  = stack_mean_tpr.std(axis=0, ddof=1) if len(valid_classes) > 1 else np.zeros_like(macro_mean_tpr)
        macro_mean_auc = float(np.mean([per_class_stats[c]["mean_auc"] for c in valid_classes]))
        macro_std_auc  = float(np.std([per_class_stats[c]["mean_auc"] for c in valid_classes], ddof=1)) if len(valid_classes) > 1 else 0.0
    else:
        mean_fpr = np.linspace(0, 1, num_points)
        macro_mean_tpr = np.zeros_like(mean_fpr)
        macro_std_tpr  = np.zeros_like(mean_fpr)
        macro_mean_auc = float("nan")
        macro_std_auc  = 0.0
    micro_agg = _interpolate_mean_std(micro_curves, num_points=num_points)
    if micro_agg is None:
        micro_mean_fpr = mean_fpr
        micro_mean_tpr = np.zeros_like(mean_fpr)
        micro_std_tpr  = np.zeros_like(mean_fpr)
        micro_mean_auc = float("nan")
        micro_std_auc  = 0.0
    else:
        micro_mean_fpr, micro_mean_tpr, micro_std_tpr, micro_mean_auc, micro_std_auc = micro_agg
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
def load_data_from_patient_files(label_csv, text_dir, mel_dir, classification):
    df = pd.read_csv(label_csv).sort_values(by='patient_id')
    patient_ids = df['patient_id'].tolist()

    if classification == 'binary':
        df['label'] = df['label'].apply(lambda x: 0 if str(x).lower() == 'healthy' else 1)
        labels = torch.tensor(df['label'].values.astype(np.int64), dtype=torch.long)
        print("Label distribution (0=healthy, 1=disease):", torch.bincount(labels).tolist())
    elif classification == 'multiclass':
        df['label'] = df['label'].astype('category').cat.codes
        labels = torch.tensor(df['label'].values.astype(np.int64), dtype=torch.long)
        print("Multiclass label distribution:", torch.bincount(labels).tolist())
    else:
        raise ValueError("classification argument must be 'binary' or 'multiclass'")

    text_embeddings, mel_raw = [], []

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
        mel_raw.append(torch.tensor(mel, dtype=torch.float32).flatten())

    max_mel_len = max(m.shape[0] for m in mel_raw)
    mel_embeddings = [F.pad(m, (0, max_mel_len - m.shape[0])) for m in mel_raw]

    text_tensor = torch.stack(text_embeddings)
    mel_tensor = torch.stack(mel_embeddings)

    # NEW: feature names
    text_feature_names = [f"text_{i}" for i in range(text_tensor.shape[1])]
    mel_feature_names  = [f"mel_{i}" for i in range(mel_tensor.shape[1])]

    return text_tensor, mel_tensor, labels, text_feature_names, mel_feature_names

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
                auc_val = roc_auc_score(true.cpu(), prob_positive.cpu())
            except Exception:
                auc_val = float('nan')
            proba_for_roc = prob_positive.cpu()  # 1D
        else:
            proba_for_roc = probs.cpu()          # 2D (N,C)
            try:
                auc_val = roc_auc_score(true.cpu(), proba_for_roc.cpu(), multi_class='ovr')
            except Exception:
                auc_val = float('nan')

        acc = accuracy_score(true.cpu(), preds.cpu())
        precision, recall, f1, _ = precision_recall_fscore_support(true.cpu(), preds.cpu(), average='weighted')
        return acc, precision, recall, f1, auc_val, preds.cpu(), true.cpu(), proba_for_roc

# -------------------- Integrated Gradients for feature importance --------------------
def _ig_single_node(model, text, mel, edge_index, device, node_idx, steps=64, target_branch='text', modality='text'):
    """
    IG attributions for node_idx w.r.t. the selected branch ('text' or 'mel').
    Baseline = mean vector over nodes (more stable than zeros).
    The non-target branch stays fixed at its actual values (if present).
    """
    model.eval()
    edge_index = edge_index.to(device)

    # Prepare inputs on device
    text_dev = text.to(device) if text is not None else None
    mel_dev  = mel.to(device) if mel is not None else None

    # Baselines
    if target_branch == 'text':
        base = text.mean(dim=0, keepdim=True).repeat(text.size(0), 1).to(device)
        x_fixed_other = mel_dev
    else:
        base = mel.mean(dim=0, keepdim=True).repeat(mel.size(0), 1).to(device)
        x_fixed_other = text_dev

    # Determine predicted class
    with torch.no_grad():
        if modality == 'both':
            logits = model(text_dev, mel_dev, edge_index)
        elif modality == 'text':
            logits = model(text_dev, edge_index)
        elif modality == 'mel':
            logits = model(mel_dev, edge_index)
        else:
            raise ValueError("Invalid modality")
        target_class = int(logits[node_idx].argmax().item())

    attributions = torch.zeros_like(base[0])
    for alpha in torch.linspace(0.0, 1.0, steps, device=device):
        # Interpolate target branch at node_idx only; others = baseline
        interp_all = base.clone()
        if target_branch == 'text':
            interp_all[node_idx] = base[node_idx] + alpha * (text_dev[node_idx] - base[node_idx])
            interp_all.requires_grad_(True)
            if modality == 'both':
                out = model(interp_all, x_fixed_other, edge_index)
            else:
                out = model(interp_all, edge_index)
            target_logit = out[node_idx, target_class]
            grads = torch.autograd.grad(target_logit, interp_all, retain_graph=False, create_graph=False)[0]
            attributions += grads[node_idx]
        else:  # mel
            interp_all[node_idx] = base[node_idx] + alpha * (mel_dev[node_idx] - base[node_idx])
            interp_all.requires_grad_(True)
            if modality == 'both':
                out = model(x_fixed_other, interp_all, edge_index)
            else:
                out = model(interp_all, edge_index)
            target_logit = out[node_idx, target_class]
            grads = torch.autograd.grad(target_logit, interp_all, retain_graph=False, create_graph=False)[0]
            attributions += grads[node_idx]

    # IG = average_grad * (input - baseline) at node_idx
    if target_branch == 'text':
        delta = text_dev[node_idx] - base[node_idx]
    else:
        delta = mel_dev[node_idx] - base[node_idx]
    ig = delta * (attributions / steps)
    return ig.detach().cpu()

def compute_global_feature_importance(model, graph_data, modality, device,
                                      feature_names, steps=64, sample_count=0,
                                      target_branch='text'):
    """
    Returns a sorted DataFrame with columns: feature, importance
    """
    if target_branch == 'text':
        X = graph_data.text
    else:
        X = graph_data.mel
    N = X.size(0)

    # Choose nodes to explain
    if (sample_count is None) or (sample_count <= 0) or (sample_count >= N):
        node_indices = list(range(N))
    else:
        rng = np.random.default_rng(42)
        node_indices = rng.choice(N, size=sample_count, replace=False).tolist()

    agg = torch.zeros(X.size(1))
    for idx in tqdm(node_indices, desc=f"Explaining nodes (IG: {target_branch})"):
        ig = _ig_single_node(
            model,
            graph_data.text if graph_data.text is not None else None,
            graph_data.mel if graph_data.mel is not None else None,
            graph_data.edge_index,
            device=device,
            node_idx=idx,
            steps=steps,
            target_branch=target_branch,
            modality=modality
        )
        agg += ig.abs()

    agg = agg / max(1, len(node_indices))
    df_imp = pd.DataFrame({"feature": feature_names, "importance": agg.numpy()})
    df_imp = df_imp.sort_values(by="importance", ascending=False).reset_index(drop=True)
    return df_imp

def save_barplot(df_imp, out_png, topk=20, title="Global Feature Importance (IG)"):
    top = df_imp.head(topk).iloc[::-1]
    plt.figure(figsize=(8, max(4, 0.35 * len(top))))
    plt.barh(top["feature"], top["importance"])
    plt.title(title)
    plt.xlabel("Attribution (|Integrated Gradients|)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# -------------------- Main --------------------
def main(args):
    # Load graph
    if not os.path.exists(args.graph_file):
        raise FileNotFoundError(f"Graph file not found: {args.graph_file}")
    graph_data = torch.load(args.graph_file, weights_only=False)

    # Outputs
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    slug = _run_slug(args)
    summary_path = results_dir / f"summary_{slug}.tsv"
    roc_path = results_dir / f"roc_{slug}.npz"
    rawjson_path = results_dir / f"raw_{slug}.json"

    # Load features
    text_embeddings, mel_embeddings, labels, text_feature_names, mel_feature_names = load_data_from_patient_files(
        args.label_csv, args.text_dir, args.mel_dir, args.classification
    )
    graph_data.text = text_embeddings
    graph_data.mel = mel_embeddings
    graph_data.y = labels

    input_dim_text = text_embeddings.shape[1]
    input_dim_mel  = mel_embeddings.shape[1]

    # ---- KFold (don’t rely on graph_data.x which may not exist) ----
    N = labels.shape[0]
    kf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
    split_iter = kf.split(np.zeros(N), labels.cpu().numpy())

    metrics = {'acc': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

    # ROC collectors
    fold_rocs_bin = []
    mc_fold_true = []
    mc_fold_proba = []
    n_classes_seen = None

    trained_model = None  # keep last fold model for explainability

    for fold, (train_idx, val_idx) in enumerate(split_iter):
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

        acc, precision, recall, f1, auc_val, preds, true, proba = evaluate(
            model, graph_data, val_idx, labels, args.modality, args.device, args.classification
        )
        print(f"[Fold {fold+1}] Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc_val:.4f}")

        cm = confusion_matrix(true, preds)
        print(f"Confusion Matrix for Fold {fold+1}:\n{cm}")

        metrics['acc'].append(acc)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        metrics['auc'].append(auc_val)

        if args.classification == 'binary':
            y_true_np = true.numpy()
            y_score_np = proba.numpy()
            fpr, tpr, _ = roc_curve(y_true_np, y_score_np)
            fold_rocs_bin.append((fpr, tpr))
        else:
            y_true_np = true.numpy()
            y_proba_np = proba.numpy()
            mc_fold_true.append(y_true_np)
            mc_fold_proba.append(y_proba_np)
            if n_classes_seen is None:
                n_classes_seen = y_proba_np.shape[1]

        trained_model = model  # keep last fold model

    # ------- Summary metrics (mean ± std) -------
    means = {k: float(np.mean(v)) if len(v) else float("nan") for k, v in metrics.items()}
    stds  = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in metrics.items()}

    print("\n=== Final Results ===")
    for k in ['acc', 'precision', 'recall', 'f1', 'auc']:
        print(f"{k.capitalize()}: {means[k]:.4f} ± {stds[k]:.4f}")

    # Save summary TSV
    df = pd.DataFrame([{
        "arch": args.arch,
        "modality": args.modality,
        "classification": args.classification,
        "graph_file": args.graph_file,
        "text_dir": args.text_dir,
        **{f"{k}_mean": means[k] for k in metrics},
        **{f"{k}_std": stds[k] for k in metrics},
    }])
    df.to_csv(summary_path, sep="\t", index=False)
    print(f"[OK] Wrote summary → {summary_path}")

    # Save raw JSON (fold metrics + any ROC stash)
    raw_payload = {
        "fold_metrics": [
            {"acc": float(a), "precision": float(p), "recall": float(r),
             "f1": float(f), "auc": float(u)}
            for a, p, r, f, u in zip(metrics['acc'], metrics['precision'],
                                     metrics['recall'], metrics['f1'], metrics['auc'])
        ],
        "meta": {
            "arch": args.arch,
            "modality": args.modality,
            "classification": args.classification,
            "graph_file": args.graph_file,
            "text_dir": args.text_dir,
        }
    }

    # ------- Save mean ± std ROC (OvR if multiclass) -------
    roc_saved = False
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
            raw_payload["roc_binary_folds"] = [
                {"fpr": fpr.tolist(), "tpr": tpr.tolist()} for (fpr, tpr) in fold_rocs_bin
            ]
            roc_saved = True
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
            per_class_std_auc  = np.array([mc["per_class"][c]["std_auc"]  for c in classes], dtype=float)
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
        raw_payload["multiclass_fold_true_lengths"] = [len(x) for x in mc_fold_true]
        roc_saved = True
        print(f"[OK] Wrote multiclass OvR mean±std ROC → {roc_path}")
    else:
        print("[INFO] No ROC data available to save.")

    # Write raw payload json
    with open(rawjson_path, "w", encoding="utf-8") as f:
        json.dump(raw_payload, f)
    print(f"[OK] Wrote raw fold data → {rawjson_path}")

    # ---- Explainability (barplot via IG) ----
    if args.explain:
        # Select feature names for requested branch
        if args.explain_branch == "text":
            feat_names = text_feature_names
        elif args.explain_branch == "mel":
            feat_names = mel_feature_names
        else:
            raise ValueError("--explain_branch must be 'text' or 'mel'")

        # Guard: modality support
        if args.modality not in ("text", "mel", "both"):
            print("[WARN] Unsupported modality for explanation.")
        else:
            print(f"[INFO] Computing Integrated Gradients for branch={args.explain_branch} ...")
            df_imp = compute_global_feature_importance(
                trained_model, graph_data, args.modality, args.device,
                feature_names=feat_names,
                steps=args.ig_steps,
                sample_count=args.explain_samples,
                target_branch=args.explain_branch
            )
            fi_csv = results_dir / f"feature_importance_{slug}_{args.explain_branch}.csv"
            df_imp.to_csv(fi_csv, index=False)
            print(f"[OK] Wrote feature importances → {fi_csv}")

            fi_png = results_dir / f"feature_importance_{slug}_{args.explain_branch}.png"
            title = f"Global Feature Importance (IG) — {args.arch} {args.modality} ({args.classification}) [{args.explain_branch}]"
            save_barplot(df_imp, fi_png, topk=args.explain_topk, title=title)
            print(f"[OK] Wrote barplot → {fi_png}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, choices=['GCN', 'GAT', 'GraphSAGE'], required=True)
    parser.add_argument('--modality', type=str, choices=['text', 'mel', 'both'], required=True)
    parser.add_argument('--classification', type=str, choices=['binary', 'multiclass'], default='binary')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--graph_file', type=str, required=True, help="Path to patient_graph.pt")
    parser.add_argument('--text_dir', type=str, required=True, help="Directory with voiceXXX-info.pt")
    parser.add_argument('--mel_dir', type=str, required=True, help="Directory with voiceXXX_stacked_mel.npy")
    parser.add_argument('--label_csv', type=str, required=True, help="CSV with columns patient_id,label")

    # Outputs
    parser.add_argument('--results_dir', type=str, default='./runs', help="Where to store outputs")

    # Explainability options
    parser.add_argument('--explain', action='store_true', help="Compute feature importances (IG) and save barplot")
    parser.add_argument('--explain_branch', type=str, choices=['text','mel'], default='text',
                        help="Which input branch to attribute (for modality='both', the other is held fixed)")
    parser.add_argument('--explain_topk', type=int, default=20, help="Top-K features to show in the barplot")
    parser.add_argument('--explain_samples', type=int, default=0, help="Number of nodes to sample for IG (0=all)")
    parser.add_argument('--ig_steps', type=int, default=64, help="Number of Riemann steps for Integrated Gradients")

    args = parser.parse_args()
    set_seed(42)
    main(args)
