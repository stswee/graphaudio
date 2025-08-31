#!/usr/bin/env python3
"""
Single 2x3 figure of AUC heatmaps:
- Rows: Binary (top), Multiclass (bottom)
- Columns: Text PSN, Clinical PSN, Mel PSN
- Inside each heatmap: rows = Modality, columns = Architecture
- Color scale fixed to [0, 1]
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False

# Canonical orders / labels
MODALITY_ORDER = ["T", "C", "M", "T + M", "C + M"]
ARCH_ORDER     = ["GCN", "GAT", "GraphSAGE"]
PSN_CANON      = ["Text", "Clinical", "Mel"]
CLASS_CANON    = ["Binary", "Multiclass"]

CLASS_MAP = {
    "binary": "Binary",
    "multi": "Multiclass",
    "multiclass": "Multiclass",
}

def _find_col(df, candidates):
    cols = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in cols:
            return cols[key]
    def norm(s): return s.lower().replace(" ", "").replace("_", "")
    norm_map = {norm(c): c for c in df.columns}
    for cand in candidates:
        nkey = norm(cand)
        if nkey in norm_map:
            return norm_map[nkey]
    raise KeyError(f"Could not find any of columns {candidates} in CSV. Available: {list(df.columns)}")

def _normalize_psn(x: str) -> str:
    if x is None:
        return x
    s = str(x).strip().lower()
    if s in ("biobert", "text", "t", "text psn", "psn text"):
        return "Text"
    if s in ("clinical", "c", "clinical psn"):
        return "Clinical"
    if s in ("mel", "m", "mel psn"):
        return "Mel"
    return str(x)

def load_and_normalize(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    col_psn   = _find_col(df, ["PSN"])
    col_node  = _find_col(df, ["Node", "Modality"])
    col_arch  = _find_col(df, ["Architecture"])
    col_class = _find_col(df, ["Class", "classification"])
    col_aucm  = _find_col(df, ["AUC Mean", "AUC_mean", "AUC (mean)", "AUC"])

    df = df.rename(columns={
        col_psn: "PSN_raw",
        col_node: "Modality",
        col_arch: "Architecture",
        col_class: "Class",
        col_aucm: "AUC_mean",
    })

    df["PSN"]   = df["PSN_raw"].astype(str).map(_normalize_psn)
    df["Class"] = df["Class"].astype(str).map(lambda x: CLASS_MAP.get(x.lower().strip(), x))

    # Standardize modality text (e.g., "T+M" -> "T + M")
    df["Modality"] = df["Modality"].astype(str)
    df["Modality"] = df["Modality"].str.replace(r"\s*\+\s*", " + ", regex=True).str.strip()
    df["Modality"] = pd.Categorical(df["Modality"], MODALITY_ORDER, ordered=True)

    # Order architectures
    df["Architecture"] = pd.Categorical(df["Architecture"], ARCH_ORDER, ordered=True)

    # Keep only canonical PSNs/classes
    df = df[df["PSN"].isin(PSN_CANON)]
    df = df[df["Class"].isin(CLASS_CANON)]

    # Numeric AUC
    df["AUC_mean"] = pd.to_numeric(df["AUC_mean"], errors="coerce")

    # Drop rows missing essentials
    df = df.dropna(subset=["PSN", "Class", "Architecture", "Modality", "AUC_mean"])
    return df

def pivot_auc_modal_rows(df_sub: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot so that rows = Modality, columns = Architecture.
    """
    pvt = (df_sub.groupby(["Modality", "Architecture"])["AUC_mean"]
           .mean()
           .unstack("Architecture")
           .reindex(index=MODALITY_ORDER, columns=ARCH_ORDER))
    return pvt

def plot_grid(df: pd.DataFrame, outpath: Path, dpi: int = 300, fmt: str = "png"):
    vmin, vmax = 0.0, 1.0  # fixed color range

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    col_titles = ["Text PSN", "Clinical PSN", "Mel PSN"]
    row_titles = ["Binary", "Multiclass"]

    last_im = None

    # Row 0 = Binary, Row 1 = Multiclass
    for r, cls in enumerate(["Binary", "Multiclass"]):
        # Columns: Text, Clinical, Mel
        for c, psn in enumerate(["Text", "Clinical", "Mel"]):
            ax = axes[r, c]
            sub = df[(df["Class"] == cls) & (df["PSN"] == psn)]
            pvt = pivot_auc_modal_rows(sub)

            if HAS_SNS:
                import seaborn as sns
                hm = sns.heatmap(
                    pvt, ax=ax, cmap="viridis", vmin=vmin, vmax=vmax,
                    annot=True, fmt=".3f", linewidths=0.5, linecolor="white",
                    cbar=False
                )
                last_im = hm.collections[0] if hm.collections else last_im
                ax.set_xlabel("Architecture")
                ax.set_ylabel("Modality")
            else:
                im = ax.imshow(pvt.values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
                last_im = im
                # annotate
                for i in range(pvt.shape[0]):
                    for j in range(pvt.shape[1]):
                        val = pvt.values[i, j]
                        s = "" if (val is None or np.isnan(val)) else f"{val:.3f}"
                        ax.text(j, i, s, ha="center", va="center", fontsize=9, color="white")
                ax.set_xticks(range(pvt.shape[1]))
                ax.set_xticklabels(pvt.columns.tolist(), rotation=45, ha="right")
                ax.set_yticks(range(pvt.shape[0]))
                ax.set_yticklabels(pvt.index.tolist())
                ax.set_xlabel("Architecture")
                ax.set_ylabel("Modality")

            if r == 0:
                ax.set_title(col_titles[c], fontsize=12, pad=8)
            if c == 0:
                ax.set_ylabel(f"{row_titles[r]} • Modality")
            else:
                # keep y-label minimal on non-first columns to save space
                ax.set_ylabel("Modality")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
        cbar.set_label("AUC")

    # fig.suptitle("GNN AUC Heatmap", fontsize=14)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath.with_suffix(f".{fmt}"), dpi=dpi)
    print(f"[OK] Saved figure → {outpath.with_suffix(f'.{fmt}')}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to results CSV")
    ap.add_argument("--out", default="./figs/GNN_AUC_Heatmap", help="Output path without extension")
    ap.add_argument("--fmt", default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    df = load_and_normalize(args.csv)
    plot_grid(df, Path(args.out), dpi=args.dpi, fmt=args.fmt)

if __name__ == "__main__":
    main()
