#!/usr/bin/env python3
"""
MLP AUC heatmap:
- Rows: Modality (T, C, M, T + M, C + M)
- Columns: Class (Binary, Multiclass)  ← first column = Binary, second = Multiclass
- Cell values: AUC Mean
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

MODALITY_ORDER = ["T", "C", "M", "T + M", "C + M"]
CLASS_ORDER    = ["Binary", "Multiclass"]

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

def load_and_normalize(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    col_mod  = _find_col(df, ["Modality", "Node"])
    col_cls  = _find_col(df, ["Class", "classification"])
    col_aucm = _find_col(df, ["AUC Mean", "AUC_mean", "AUC (mean)", "AUC"])

    df = df.rename(columns={
        col_mod:  "Modality",
        col_cls:  "Class",
        col_aucm: "AUC_mean",
    })

    # Normalize
    df["Class"] = df["Class"].astype(str).map(lambda x: CLASS_MAP.get(x.lower().strip(), x))
    df["Modality"] = (df["Modality"].astype(str)
                      .str.replace(r"\s*\+\s*", " + ", regex=True)
                      .str.strip())

    # Keep desired categories and cast to ordered categoricals
    df["Class"]    = pd.Categorical(df["Class"], CLASS_ORDER, ordered=True)
    df["Modality"] = pd.Categorical(df["Modality"], MODALITY_ORDER, ordered=True)

    df["AUC_mean"] = pd.to_numeric(df["AUC_mean"], errors="coerce")
    df = df.dropna(subset=["Modality", "Class", "AUC_mean"])
    return df

def make_heatmap(df: pd.DataFrame, outpath: Path, dpi: int = 300, fmt: str = "png"):
    # Pivot to rows=Modality, cols=Class
    pvt = (df.groupby(["Modality", "Class"])["AUC_mean"]
             .mean()
             .unstack("Class")
             .reindex(index=MODALITY_ORDER, columns=CLASS_ORDER))

    vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)

    if HAS_SNS:
        import seaborn as sns
        hm = sns.heatmap(
            pvt, ax=ax, cmap="viridis", vmin=vmin, vmax=vmax,
            annot=True, fmt=".3f", linewidths=0.5, linecolor="white",
            cbar=True, cbar_kws={"shrink": 0.8}
        )
    else:
        im = ax.imshow(pvt.values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(pvt.shape[0]):
            for j in range(pvt.shape[1]):
                val = pvt.values[i, j]
                s = "" if (val is None or np.isnan(val)) else f"{val:.3f}"
                ax.text(j, i, s, ha="center", va="center", fontsize=9, color="white")
        ax.set_xticks(range(pvt.shape[1])); ax.set_xticklabels(pvt.columns.tolist())
        ax.set_yticks(range(pvt.shape[0])); ax.set_yticklabels(pvt.index.tolist())
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label("AUC")

    ax.set_xlabel("Class")
    ax.set_ylabel("Modality")
    # ax.set_title("MLP AUC Heatmap")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath.with_suffix(f".{fmt}"), dpi=dpi)
    print(f"[OK] Saved → {outpath.with_suffix(f'.{fmt}')}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to MLP results CSV (e.g., 'WQE Results - MLP.csv')")
    ap.add_argument("--out", default="./figs/mlp_auc_heatmap", help="Output path without extension")
    ap.add_argument("--fmt", default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    df = load_and_normalize(args.csv)
    make_heatmap(df, Path(args.out), dpi=args.dpi, fmt=args.fmt)

if __name__ == "__main__":
    main()
