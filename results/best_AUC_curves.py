#!/usr/bin/env python3
"""
Figure 3: Best ROC curves (GNN vs MLP)
- Panel 1 (left): Binary
- Panel 2 (right): Multiclass (macro-average)
"""

import argparse
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

def parse_label_from_path(p: Path) -> str:
    """
    Expect filenames like:
      roc_<ARCH>__<MODALITY>__<CLASS>__... .npz
    Example:
      roc_GraphSAGE__clinical__binary__patient_graph...npz
      roc_MLP__text__multiclass__...npz
    """
    stem = p.stem  # without .npz
    m = re.match(r"roc_([^_]+)__([^_]+)__([^_]+)__", stem)
    if m:
        arch, modality, clz = m.groups()
        arch = arch.replace("-", "").upper()
        modality = {"text":"Text","clinical":"Clinical","mel":"Mel","both":"Text+Mel","c+m":"Clinical+Mel","t+m":"Text+Mel"}.get(modality.lower(), modality)
        clz = "Binary" if clz.lower().startswith("bin") else "Multiclass"
        return f"{arch} ({modality})"
    return p.stem

def _clip01(x):
    return np.clip(x, 0.0, 1.0)

def load_binary_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    # Required keys from your writer: mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc
    return dict(
        fpr=d["mean_fpr"],
        tpr=d["mean_tpr"],
        std=d["std_tpr"],
        auc_mean=float(d["mean_auc"]),
        auc_std=float(d["std_auc"]),
    )

def load_multiclass_npz(path: Path, which="macro"):
    """
    For 'multiclass_ovr' files. Use macro by default.
    Keys (from your saver): macro_mean_fpr, macro_mean_tpr, macro_std_tpr, macro_mean_auc, macro_std_auc
    """
    d = np.load(path, allow_pickle=True)
    prefix = "macro" if which == "macro" else "micro"
    return dict(
        fpr=d[f"{prefix}_mean_fpr"],
        tpr=d[f"{prefix}_mean_tpr"],
        std=d[f"{prefix}_std_tpr"],
        auc_mean=float(d[f"{prefix}_mean_auc"]),
        auc_std=float(d[f"{prefix}_std_auc"]),
    )

def plot_roc(ax, curves, title: str):
    ax.plot([0,1], [0,1], linestyle="--", linewidth=1, color="gray", alpha=0.7, label="Chance")
    for c in curves:
        fpr = c["fpr"]
        tpr = c["tpr"]
        std = c["std"]
        label = c["label"]
        auc_mean = c["auc_mean"]
        auc_std = c["auc_std"]

        ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC {auc_mean:.3f}±{auc_std:.3f})")
        if std is not None:
            tpr_upper = _clip01(tpr + std)
            tpr_lower = _clip01(tpr - std)
            ax.fill_between(fpr, tpr_lower, tpr_upper, alpha=0.2)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, frameon=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin_gnn", required=True, help="NPZ for best *GNN* binary ROC")
    ap.add_argument("--bin_mlp", required=True, help="NPZ for best *MLP* binary ROC")
    ap.add_argument("--mc_gnn",  required=True, help="NPZ for best *GNN* multiclass ROC")
    ap.add_argument("--mc_mlp",  required=True, help="NPZ for best *MLP* multiclass ROC")
    ap.add_argument("--out", default="./figs/figure3_best_roc", help="Output path without extension")
    ap.add_argument("--fmt", default="png", choices=["png","pdf","svg"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--avg", default="macro", choices=["macro","micro"], help="Multiclass average to plot")
    args = ap.parse_args()

    bin_gnn_path = Path(args.bin_gnn)
    bin_mlp_path = Path(args.bin_mlp)
    mc_gnn_path  = Path(args.mc_gnn)
    mc_mlp_path  = Path(args.mc_mlp)

    # Load curves
    bin_gnn = load_binary_npz(bin_gnn_path); bin_gnn["label"] = parse_label_from_path(bin_gnn_path)
    bin_mlp = load_binary_npz(bin_mlp_path); bin_mlp["label"] = parse_label_from_path(bin_mlp_path)
    mc_gnn  = load_multiclass_npz(mc_gnn_path, which=args.avg); mc_gnn["label"]  = parse_label_from_path(mc_gnn_path)
    mc_mlp  = load_multiclass_npz(mc_mlp_path, which=args.avg); mc_mlp["label"]  = parse_label_from_path(mc_mlp_path)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    plot_roc(axes[0], [bin_gnn, bin_mlp], title="Binary ROC (Best GNN vs MLP)")
    plot_roc(axes[1], [mc_gnn, mc_mlp],   title=f"Multiclass ROC ({args.avg.title()} Avg) (Best GNN vs MLP)")

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    outfile = out_base.with_suffix(f".{args.fmt}")
    fig.savefig(outfile, dpi=args.dpi)
    print(f"[OK] Saved Figure 3 → {outfile}")

if __name__ == "__main__":
    main()
