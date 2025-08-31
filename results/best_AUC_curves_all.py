#!/usr/bin/env python3
"""
Figure 3: Best ROC curves (GNN vs MLP)
- Panel 1 (left): Binary
- Panel 2 (middle): Multiclass (macro-average)
- Panel 3 (right): Multiclass (per-class curves with condition names)
"""

import argparse
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Class-name mapping (from your provided CSV)
# If the NPZ 'classes' array already contains strings, we use those.
# If it contains numeric IDs, we'll map with this dict.
CLASS_NAMES = {
    0: "healthy",
    1: "hyperkinetic dysphonia",
    2: "hypokinetic dysphonia",
    3: "reflux laryngitis",
}
# ---------------------------------------------------------

def parse_label_from_path(p: Path) -> str:
    """
    Expect filenames like:
      roc_<ARCH>__<MODALITY>__<CLASS>__... .npz
    Examples:
      roc_GraphSAGE__clinical__binary__patient_graph...npz
      roc_MLP__text__multiclass__...npz
    Returns a concise label, e.g., "GRAPHSAGE (Text)"
    """
    stem = p.stem  # without .npz
    m = re.match(r"roc_([^_]+)__([^_]+)__([^_]+)__", stem)
    if m:
        arch, modality, clz = m.groups()
        arch = arch.replace("-", "").upper()
        modality = {
            "text": "Text",
            "clinical": "Clinical",
            "mel": "Mel",
            "both": "Text+Mel",
            "c+m": "Clinical+Mel",
            "t+m": "Text+Mel"
        }.get(modality.lower(), modality)
        return f"{arch} ({modality})"
    return p.stem

def _clip01(x):
    return np.clip(x, 0.0, 1.0)

# ===================== Loaders =====================

def load_binary_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    # Required keys: mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc
    return dict(
        fpr=d["mean_fpr"],
        tpr=d["mean_tpr"],
        std=d.get("std_tpr", None),
        auc_mean=float(d["mean_auc"]),
        auc_std=float(d["std_auc"]),
    )

def load_multiclass_avg_npz(path: Path, which="macro"):
    """
    Multiclass macro/micro average.
    Keys expected:
      - macro_mean_fpr, macro_mean_tpr, macro_std_tpr, macro_mean_auc, macro_std_auc
      - (and/or micro_* equivalents)
    """
    d = np.load(path, allow_pickle=True)
    prefix = "macro" if which == "macro" else "micro"
    return dict(
        fpr=d[f"{prefix}_mean_fpr"],
        tpr=d[f"{prefix}_mean_tpr"],
        std=d.get(f"{prefix}_std_tpr", None),
        auc_mean=float(d[f"{prefix}_mean_auc"]),
        auc_std=float(d[f"{prefix}_std_auc"]),
    )

def load_multiclass_per_class_npz(path: Path):
    """
    Per-class content.
    Expected keys:
      - mean_fpr
      - per_class_mean_tpr (shape: [len(mean_fpr), n_classes] or [n_classes, len(mean_fpr)])
      - per_class_std_tpr  (optional, same shape as mean_tpr)
      - per_class_mean_auc (shape: [n_classes])
      - per_class_std_auc  (shape: [n_classes])
      - classes            (shape: [n_classes])  # may be ints or strings
    """
    d = np.load(path, allow_pickle=True)
    mean_fpr = d["mean_fpr"]
    per_class_mean_tpr = d["per_class_mean_tpr"]
    per_class_std_tpr = d.get("per_class_std_tpr", None)
    per_class_mean_auc = d["per_class_mean_auc"]
    per_class_std_auc = d["per_class_std_auc"]
    classes = d["classes"]

    # Fix orientation if needed
    if per_class_mean_tpr.shape[0] != mean_fpr.shape[0] and per_class_mean_tpr.shape[1] == mean_fpr.shape[0]:
        per_class_mean_tpr = per_class_mean_tpr.T
        if per_class_std_tpr is not None and per_class_std_tpr.shape != per_class_mean_tpr.shape:
            per_class_std_tpr = per_class_std_tpr.T

    n_points, _ = per_class_mean_tpr.shape
    assert mean_fpr.shape[0] == n_points, "mean_fpr length must match TPR length per class"

    # Normalize class entries to Python str or int where possible
    norm_classes = []
    for c in classes:
        # c may be np.str_, np.int_, bytes, etc.
        try:
            # If it's a clean digit-like string, keep as int; else use str
            s = str(c)
            if s.isdigit():
                norm_classes.append(int(s))
            else:
                norm_classes.append(s)
        except Exception:
            norm_classes.append(str(c))

    return dict(
        fpr=mean_fpr,
        per_class_tpr=per_class_mean_tpr,
        per_class_std=per_class_std_tpr,
        per_class_auc_mean=per_class_mean_auc.astype(float),
        per_class_auc_std=per_class_std_auc.astype(float),
        classes=norm_classes,  # list of ints or strings
    )

# ===================== Plotters =====================

def plot_roc(ax, curves, title: str):
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", alpha=0.7, label="Chance")
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
    ax.legend(loc="lower right", fontsize=6, frameon=True)

def _class_display_name(idx: int, raw_entry):
    """
    Decide what to show in the legend for class idx.
    - If raw_entry is a non-numeric string (e.g., 'healthy'), use it directly.
    - If raw_entry is numeric (int or digit string), map via CLASS_NAMES if possible.
    - Else fallback to str(raw_entry).
    """
    # Non-numeric strings (condition names) get used as-is
    if isinstance(raw_entry, str) and not raw_entry.isdigit():
        return raw_entry

    # Numeric classes -> use our mapping if available
    try:
        key = int(raw_entry) if isinstance(raw_entry, str) else int(raw_entry)
        return CLASS_NAMES.get(key, str(raw_entry))
    except Exception:
        # Fallbacks: try idx -> name, else raw_entry
        return CLASS_NAMES.get(idx, str(raw_entry))

def plot_roc_per_class(ax, gnn, mlp, gnn_label_base: str, mlp_label_base: str, title: str):
    """
    Plot per-class ROC curves for GNN (solid) and MLP (dashed).
    One color per class to compare models on the same class.
    """
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", alpha=0.7, label="Chance")

    fpr = gnn["fpr"]
    classes_g = gnn["classes"]
    n_classes = len(classes_g)

    # Safety checks
    assert np.allclose(fpr, mlp["fpr"]), "GNN and MLP should share the same mean_fpr grid"
    classes_m = mlp["classes"]
    assert len(classes_m) == n_classes, "Class list lengths must match between GNN and MLP"

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)
    if not color_cycle:
        color_cycle = [None] * n_classes  # let matplotlib choose defaults

    for idx in range(n_classes):
        raw_g = classes_g[idx]
        raw_m = classes_m[idx]
        # Display name based on GNN's entry (they should align)
        cls_name = _class_display_name(idx, raw_g)

        color = color_cycle[idx % len(color_cycle)] if color_cycle else None

        # GNN (solid)
        tpr_g = gnn["per_class_tpr"][:, idx]
        std_g = None if gnn["per_class_std"] is None else gnn["per_class_std"][:, idx]
        auc_g = float(gnn["per_class_auc_mean"][idx])
        auc_gs = float(gnn["per_class_auc_std"][idx])
        ax.plot(fpr, tpr_g, linewidth=2, linestyle="-", color=color,
                label=f"{gnn_label_base}/{cls_name} (AUC {auc_g:.3f}±{auc_gs:.3f})")
        if std_g is not None:
            ax.fill_between(fpr, _clip01(tpr_g - std_g), _clip01(tpr_g + std_g),
                            alpha=0.12, color=color)

        # MLP (dashed)
        tpr_m = mlp["per_class_tpr"][:, idx]
        std_m = None if mlp["per_class_std"] is None else mlp["per_class_std"][:, idx]
        auc_m = float(mlp["per_class_auc_mean"][idx])
        auc_ms = float(mlp["per_class_auc_std"][idx])
        ax.plot(fpr, tpr_m, linewidth=2, linestyle="--", color=color,
                label=f"{mlp_label_base}/{cls_name} (AUC {auc_m:.3f}±{auc_ms:.3f})")
        if std_m is not None:
            ax.fill_between(fpr, _clip01(tpr_m - std_m), _clip01(tpr_m + std_m),
                            alpha=0.12, color=color)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=6, frameon=True, ncol=1)

# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin_gnn", required=True, help="NPZ for best *GNN* binary ROC")
    ap.add_argument("--bin_mlp", required=True, help="NPZ for best *MLP* binary ROC")
    ap.add_argument("--mc_gnn",  required=True, help="NPZ for best *GNN* multiclass ROC")
    ap.add_argument("--mc_mlp",  required=True, help="NPZ for best *MLP* multiclass ROC")
    ap.add_argument("--out", default="./figs/figure3_best_roc", help="Output path without extension")
    ap.add_argument("--fmt", default="png", choices=["png","pdf","svg"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--avg", default="macro", choices=["macro","micro"], help="Multiclass average for Panel 2")
    args = ap.parse_args()

    bin_gnn_path = Path(args.bin_gnn)
    bin_mlp_path = Path(args.bin_mlp)
    mc_gnn_path  = Path(args.mc_gnn)
    mc_mlp_path  = Path(args.mc_mlp)

    # Load curves
    bin_gnn = load_binary_npz(bin_gnn_path); bin_gnn["label"] = parse_label_from_path(bin_gnn_path)
    bin_mlp = load_binary_npz(bin_mlp_path); bin_mlp["label"] = parse_label_from_path(bin_mlp_path)

    mc_gnn_avg = load_multiclass_avg_npz(mc_gnn_path, which=args.avg); mc_gnn_avg["label"] = parse_label_from_path(mc_gnn_path)
    mc_mlp_avg = load_multiclass_avg_npz(mc_mlp_path, which=args.avg); mc_mlp_avg["label"] = parse_label_from_path(mc_mlp_path)

    mc_gnn_pc = load_multiclass_per_class_npz(mc_gnn_path)
    mc_mlp_pc = load_multiclass_per_class_npz(mc_mlp_path)
    gnn_label_base = parse_label_from_path(mc_gnn_path)
    mlp_label_base = parse_label_from_path(mc_mlp_path)

    # Figure (1x3)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # Panel 1: Binary ROC (Best GNN vs MLP)
    plot_roc(axes[0], [bin_gnn, bin_mlp], title="Binary ROC (Best GNN vs MLP)")

    # Panel 2: Multiclass ROC (Macro-/Micro-average, Best GNN vs MLP)
    title_avg = f"Multiclass ROC — {args.avg.title()} Avg (Best GNN vs MLP)"
    plot_roc(axes[1], [mc_gnn_avg, mc_mlp_avg], title=title_avg)

    # Panel 3: Multiclass ROC (Per-class, Best GNN vs MLP) with condition names
    plot_roc_per_class(
        axes[2],
        mc_gnn_pc,
        mc_mlp_pc,
        gnn_label_base=gnn_label_base,
        mlp_label_base=mlp_label_base,
        title="Multiclass ROC — Per-class (Best GNN vs MLP)"
    )

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    outfile = out_base.with_suffix(f".{args.fmt}")
    fig.savefig(outfile, dpi=args.dpi)
    print(f"[OK] Saved Figure 3 → {outfile}")

if __name__ == "__main__":
    main()
