from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXCLUDE_COLS = {"smiles", "id", "name", "cas", "dup_count", "rdkit_valid"}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower()
    return df


def _replace_inf_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)


def _load_split_frames(project_root: Path) -> pd.DataFrame:
    processed_dir = project_root / "data" / "processed"
    paths = [
        processed_dir / "train_data.csv",
        processed_dir / "val_data.csv",
        processed_dir / "test_data.csv",
    ]
    frames: List[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing split CSV: {path}")
        frames.append(pd.read_csv(path))
    merged = pd.concat(frames, axis=0, ignore_index=True)
    return merged


def _select_feature_columns(
    df: pd.DataFrame, target_col: str, feature_set: str
) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [
        c for c in numeric_cols if c not in EXCLUDE_COLS and c != target_col
    ]

    if feature_set == "rdkit":
        return [c for c in numeric_cols if c.startswith("rdkit_")]
    if feature_set == "cdft":
        return [c for c in numeric_cols if not c.startswith("rdkit_")]
    return numeric_cols


def _drop_constant_cols(df: pd.DataFrame) -> pd.DataFrame:
    nunique = df.nunique(dropna=False)
    keep = [c for c in df.columns if int(nunique.get(c, 0)) > 1]
    return df[keep]


def _apply_academic_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.linewidth": 1.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 1.4,
            "ytick.major.width": 1.4,
            "xtick.minor.width": 1.2,
            "ytick.minor.width": 1.2,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot global Spearman correlation heatmap for Top-K features."
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="log(1o2)",
        help="Target column name (required, normalized to lowercase).",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="both",
        choices=["cdft", "rdkit", "both"],
        help="Feature subset to use: cdft, rdkit, or both.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of top features by |rho| to visualize (default: 30).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/figures",
        help="Output directory for heatmap files.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    df = _load_split_frames(project_root)
    df = _replace_inf_with_nan(_normalize_columns(df))

    target_col = args.target_col.strip().lower()
    if target_col not in df.columns:
        cols = ", ".join(sorted(df.columns))
        raise ValueError(
            f"Target column {target_col!r} not found. Available columns: {cols}"
        )

    feature_cols = _select_feature_columns(df, target_col, args.feature_set)
    if not feature_cols:
        raise ValueError("No numeric feature columns found after filtering.")

    X = _drop_constant_cols(df[feature_cols])
    if X.shape[1] == 0:
        raise ValueError("No non-constant feature columns available.")

    y = df[target_col]
    rho = X.corrwith(y, method="spearman")
    rho = rho.dropna()
    if rho.empty:
        raise ValueError("No valid Spearman correlations computed.")

    top_k = int(args.top_k)
    if top_k <= 0:
        raise ValueError("--top-k must be > 0")
    rho_sorted = rho.reindex(rho.abs().sort_values(ascending=False).index)
    rho_top = rho_sorted.head(top_k)
    top_features = rho_top.index.tolist()

    corr = X[top_features].corr(method="spearman")
    corr = corr.loc[top_features, top_features]

    _apply_academic_style()
    fig, ax = plt.subplots(figsize=(10, 9), dpi=300)
    im = ax.imshow(
        corr.to_numpy(),
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
        interpolation="nearest",
    )
    ax.set_xticks(range(len(top_features)))
    ax.set_yticks(range(len(top_features)))
    ax.set_xticklabels(top_features, rotation=90, fontsize=7)
    ax.set_yticklabels(top_features, fontsize=7)
    ax.tick_params(axis="both", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ")

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.6)

    out_dir = Path(args.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else (project_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_tag = args.feature_set.strip().upper()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"Spearman_Heatmap_{feat_tag}_Top{top_k}_{stamp}"
    out_png = out_dir / f"{stem}.png"
    out_csv = out_dir / f"TopFeatures_Spearman_{feat_tag}_Top{top_k}_{stamp}.csv"

    top_table = pd.DataFrame(
        {
            "feature": rho_top.index,
            "spearman_rho": rho_top.values,
            "abs_rho": rho_top.abs().values,
        }
    )
    top_table.to_csv(out_csv, index=False, encoding="utf-8-sig")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")

    print("Input CSVs:", "train_data.csv + val_data.csv + test_data.csv")
    print("Top-K features:", len(top_features))
    print("Output PNG:", out_png)
    print("Output CSV:", out_csv)


if __name__ == "__main__":
    main()
