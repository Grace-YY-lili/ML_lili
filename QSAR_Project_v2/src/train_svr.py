from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from run_artifacts import (
    load_split_manifest,
    normalize_columns,
    replace_inf_with_nan,
    resolve_target_col,
    select_feature_columns,
    split_seed_from_manifest,
)


DEFAULT_C = 2.0914981329
DEFAULT_EPSILON = 0.0805095587
DEFAULT_GAMMA = 0.0137853223
DEFAULT_TARGET_COL = "log(1o2)"

FEATSET_TAGS = {
    "cdft": "CDFT",
    "rdkit": "RDKIT",
    "both": "BOTH",
}


def _split_seed_from_manifest(project_root: Path) -> str:
    manifest = load_split_manifest(project_root)
    return split_seed_from_manifest(manifest)


def _prepare_xy(
    df: pd.DataFrame, feature_cols: List[str], target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[feature_cols]
    y = df[target_col]
    return X, y


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
    }


def _build_run_dir(project_root: Path, seed: int, feature_set: str) -> Path:
    date_str = datetime.now().strftime("%Y-%m-%d")
    split_seed = _split_seed_from_manifest(project_root)
    feat_tag = FEATSET_TAGS.get(feature_set, "RAW")
    run_name = f"{date_str}_svr_seed{seed}_split{split_seed}_feats{feat_tag}"
    run_dir = project_root / "results" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SVR baseline on prepared splits.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed label for run naming.")
    parser.add_argument("--C", type=float, default=DEFAULT_C, help="SVR C parameter.")
    parser.add_argument(
        "--epsilon", type=float, default=DEFAULT_EPSILON, help="SVR epsilon parameter."
    )
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="SVR gamma parameter.")
    parser.add_argument(
        "--target-col",
        type=str,
        default=DEFAULT_TARGET_COL,
        help="Target column name (required; default: log(1o2)).",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=["cdft", "rdkit", "both"],
        default="both",
        help="Feature subset to use: cdft, rdkit, or both.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"

    train_path = processed_dir / "train_data.csv"
    val_path = processed_dir / "val_data.csv"
    test_path = processed_dir / "test_data.csv"

    train_df = normalize_columns(replace_inf_with_nan(pd.read_csv(train_path)))
    val_df = normalize_columns(replace_inf_with_nan(pd.read_csv(val_path)))
    test_df = normalize_columns(replace_inf_with_nan(pd.read_csv(test_path)))

    target_col = resolve_target_col(train_df, args.target_col)
    feature_cols = select_feature_columns(train_df, target_col, args.feature_set)
    if not feature_cols:
        raise ValueError("No numeric feature columns found after excluding the target column.")

    X_train, y_train = _prepare_xy(train_df, feature_cols, target_col)
    X_val, y_val = _prepare_xy(val_df, feature_cols, target_col)
    X_test, y_test = _prepare_xy(test_df, feature_cols, target_col)

    # Train-only preprocessing
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imputed = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imputed)

    X_val_scaled = scaler.transform(imputer.transform(X_val))
    X_test_scaled = scaler.transform(imputer.transform(X_test))

    model = SVR(C=args.C, epsilon=args.epsilon, gamma=args.gamma)
    model.fit(X_train_scaled, y_train.to_numpy())

    val_pred = model.predict(X_val_scaled)
    test_pred = model.predict(X_test_scaled)

    metrics = {
        "val": _compute_metrics(y_val.to_numpy(), val_pred),
        "test": _compute_metrics(y_test.to_numpy(), test_pred),
        "n": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "target_col": target_col,
        "feature_count": int(len(feature_cols)),
        "feature_set": args.feature_set,
        "split_seed": _split_seed_from_manifest(project_root),
    }

    run_dir = _build_run_dir(project_root, args.seed, args.feature_set)

    params_path = run_dir / "params.json"
    metrics_path = run_dir / "metrics.json"
    pred_val_path = run_dir / "predictions_val.csv"
    pred_test_path = run_dir / "predictions_test.csv"
    model_path = run_dir / "model.pkl"

    params_payload = {
        "model": "SVR",
        "seed": args.seed,
        "C": args.C,
        "epsilon": args.epsilon,
        "gamma": args.gamma,
        "target_col_requested": args.target_col,
        "target_col_resolved": target_col,
        "feature_set": args.feature_set,
        "feature_columns": feature_cols,
        "split_seed": _split_seed_from_manifest(project_root),
    }

    params_path.write_text(json.dumps(params_payload, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    val_out = pd.DataFrame(
        {
            "y_true": y_val.to_numpy(),
            "y_pred": val_pred,
        }
    )
    test_out = pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "y_pred": test_pred,
        }
    )

    # Preserve SMILES if present for traceability.
    if "smiles" in val_df.columns:
        val_out.insert(0, "smiles", val_df["smiles"])
    if "smiles" in test_df.columns:
        test_out.insert(0, "smiles", test_df["smiles"])

    val_out.to_csv(pred_val_path, index=False)
    test_out.to_csv(pred_test_path, index=False)

    with model_path.open("wb") as f:
        pickle.dump(
            {
                "imputer": imputer,
                "scaler": scaler,
                "model": model,
                "feature_columns": feature_cols,
                "target_column": target_col,
                "feature_set": args.feature_set,
            },
            f,
        )

    print("Run directory:")
    print(run_dir)
    print("Validation metrics:", metrics["val"])
    print("Test metrics:", metrics["test"])


if __name__ == "__main__":
    main()
