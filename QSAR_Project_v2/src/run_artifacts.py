from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


DEFAULT_C = 2.0914981329
DEFAULT_GAMMA = 0.0137853223
DEFAULT_EPSILON = 0.0805095587

NON_FEATURE_NUMERIC_COLS = {"rdkit_valid"}


def load_split_manifest(project_root: Path) -> Dict[str, Any]:
    manifest_path = project_root / "data" / "processed" / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing split manifest at {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse split manifest at {manifest_path}") from exc
    if "effective_split_seed" not in manifest and "split_seed" not in manifest:
        raise ValueError("split_manifest.json missing effective_split_seed/split_seed.")
    return manifest


def split_seed_from_manifest(manifest: Dict[str, Any]) -> str:
    if "effective_split_seed" in manifest:
        return str(manifest["effective_split_seed"])
    if "split_seed" in manifest:
        return str(manifest["split_seed"])
    raise ValueError("split manifest missing effective_split_seed/split_seed.")


def resolve_run_dir(project_root: Path, run_dir_arg: str) -> Path:
    p = Path(run_dir_arg)
    resolved = p if p.is_absolute() else (project_root / p).resolve()
    if resolved.exists():
        return resolved

    leaf = p.name
    archive_root = project_root / "results" / "archive"
    if archive_root.exists():
        candidates = [d for d in archive_root.rglob(leaf) if d.is_dir()]
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            return max(candidates, key=lambda d: d.stat().st_mtime)

    return resolved


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower()
    return df


def replace_inf_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)


def resolve_target_col(df: pd.DataFrame, target_col: str) -> str:
    target = target_col.strip().lower()
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found in columns: {list(df.columns)}")
    return target


def select_feature_columns(df: pd.DataFrame, target_col: str, feature_set: str) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    rdkit_cols = [c for c in numeric_cols if c.startswith("rdkit_")]
    exclude = {target_col, *NON_FEATURE_NUMERIC_COLS}

    if feature_set == "cdft":
        return [c for c in numeric_cols if c not in exclude and not c.startswith("rdkit_")]
    if feature_set == "rdkit":
        return [c for c in rdkit_cols if c not in exclude]
    return [c for c in numeric_cols if c not in exclude]


def _get_pred_cols(df: pd.DataFrame) -> Tuple[str, str]:
    cols = [c.strip().lower() for c in df.columns]
    mapping = {c.strip().lower(): c for c in df.columns}
    if "y_true" in cols and "y_pred" in cols:
        return mapping["y_true"], mapping["y_pred"]
    if "true" in cols and "pred" in cols:
        return mapping["true"], mapping["pred"]
    if "y" in cols and "yhat" in cols:
        return mapping["y"], mapping["yhat"]
    raise ValueError("predictions file must contain y_true/y_pred columns.")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
    }


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_target_from_params(params: Dict[str, Any], fallback: str | None) -> str:
    for key in ("target_col", "target_col_resolved", "target_col_requested"):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    if fallback:
        return fallback.strip().lower()
    raise ValueError("Target column not found in params.json and no fallback provided.")


def _resolve_feature_set_from_params(params: Dict[str, Any], fallback: str | None) -> str:
    value = params.get("feature_set")
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    if fallback:
        return fallback.strip().lower()
    raise ValueError("Feature set not found in params.json and no fallback provided.")


def _resolve_feature_columns(
    params: Dict[str, Any],
    df: pd.DataFrame,
    target_col: str,
    feature_set: str,
) -> List[str]:
    cols = params.get("feature_columns")
    if isinstance(cols, list) and cols:
        return [str(c).strip().lower() for c in cols]
    return select_feature_columns(df, target_col, feature_set)


def _resolve_svr_params(
    params: Dict[str, Any],
    selection: Dict[str, Any],
) -> Tuple[float, float, float]:
    chosen = selection.get("chosen_params")
    if isinstance(chosen, dict):
        try:
            chosen_tuple = (float(chosen["C"]), float(chosen["gamma"]), float(chosen["epsilon"]))
        except Exception:
            chosen_tuple = None
    else:
        chosen_tuple = None

    best = params.get("best_params")
    if isinstance(best, dict):
        try:
            best_tuple = (float(best["C"]), float(best["gamma"]), float(best["epsilon"]))
        except Exception:
            best_tuple = None
    else:
        best_tuple = None

    if chosen_tuple and best_tuple and chosen_tuple != best_tuple:
        raise ValueError(
            f"Params mismatch: selection.json {chosen_tuple} != params.json {best_tuple}"
        )
    if chosen_tuple:
        return chosen_tuple
    if best_tuple:
        return best_tuple

    try:
        return float(params["C"]), float(params["gamma"]), float(params["epsilon"])
    except Exception:
        return DEFAULT_C, DEFAULT_GAMMA, DEFAULT_EPSILON


def _apply_outlier_filter(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    outlier_z: float,
    outlier_on: str,
) -> Tuple[pd.DataFrame, np.ndarray, int]:
    if outlier_z <= 0:
        return X_train, y_train, 0

    if outlier_on == "y":
        mean = float(np.mean(y_train))
        std = float(np.std(y_train, ddof=0))
        if std == 0:
            return X_train, y_train, 0
        z = (y_train - mean) / std
        mask = np.abs(z) <= outlier_z
    else:
        imputer = SimpleImputer(strategy="median")
        X_imp = imputer.fit_transform(X_train)
        mean = np.mean(X_imp, axis=0)
        std = np.std(X_imp, axis=0, ddof=0)
        std[std == 0] = 1.0
        z = (X_imp - mean) / std
        mask = (np.abs(z) <= outlier_z).all(axis=1)

    removed = int((~mask).sum())
    return X_train.loc[mask].reset_index(drop=True), y_train[mask], removed


def _write_metrics_if_changed(path: Path, payload: Dict[str, Any]) -> None:
    existing = _load_json(path)
    merged = dict(existing)
    merged.update(payload)
    text = json.dumps(merged, indent=2)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")


def load_or_write_predictions_and_metrics(
    run_dir: Path,
    mode: str = "test",
    project_root: Path | None = None,
    target_col: str | None = None,
    feature_set: str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if mode not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported mode: {mode!r}. Use 'train', 'val' or 'test'.")

    run_dir = Path(run_dir)
    manifest = load_split_manifest(project_root) if project_root else None
    pred_path = run_dir / f"predictions_{mode}.csv"
    metrics_path = run_dir / "metrics.json"

    if pred_path.exists():
        if manifest is not None:
            params = _load_json(run_dir / "params.json")
            params_seed = params.get("split_seed")
            if params_seed is not None:
                manifest_seed = split_seed_from_manifest(manifest)
                if str(params_seed) != str(manifest_seed):
                    raise ValueError(
                        f"Split seed mismatch: run expects {params_seed}, manifest has {manifest_seed}."
                    )
        pred_df = pd.read_csv(pred_path)
        y_true_col, y_pred_col = _get_pred_cols(pred_df)
        metrics = _compute_metrics(
            pred_df[y_true_col].to_numpy(),
            pred_df[y_pred_col].to_numpy(),
        )

        payload = _load_json(metrics_path)
        payload[mode] = metrics
        _write_metrics_if_changed(metrics_path, payload)
        return pred_df, payload

    if project_root is None:
        raise ValueError("project_root is required to rebuild predictions.")

    params_path = run_dir / "params.json"
    selection_path = run_dir / "selection.json"
    params = _load_json(params_path)
    selection = _load_json(selection_path)

    manifest_seed = split_seed_from_manifest(manifest)
    params_seed = params.get("split_seed")
    if params_seed is not None and str(params_seed) != str(manifest_seed):
        raise ValueError(
            f"Split seed mismatch: run expects {params_seed}, manifest has {manifest_seed}."
        )

    resolved_target = _resolve_target_from_params(params, target_col)
    resolved_feature_set = _resolve_feature_set_from_params(params, feature_set)

    processed_dir = project_root / "data" / "processed"
    train_path = processed_dir / "train_data.csv"
    val_path = processed_dir / "val_data.csv"
    test_path = processed_dir / "test_data.csv"

    train_df = replace_inf_with_nan(normalize_columns(pd.read_csv(train_path)))
    val_df = replace_inf_with_nan(normalize_columns(pd.read_csv(val_path)))
    test_df = replace_inf_with_nan(normalize_columns(pd.read_csv(test_path)))

    target = resolve_target_col(train_df, resolved_target)
    feature_cols = _resolve_feature_columns(params, train_df, target, resolved_feature_set)

    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"Feature columns missing from data: {missing}")

    X_train = train_df[feature_cols]
    y_train = train_df[target].to_numpy()
    X_val = val_df[feature_cols]
    y_val = val_df[target].to_numpy()
    X_test = test_df[feature_cols]
    y_test = test_df[target].to_numpy()

    outlier_z = float(params.get("outlier_z", 0.0) or 0.0)
    outlier_on = str(params.get("outlier_on", "y"))
    if outlier_z > 0:
        X_train, y_train, _ = _apply_outlier_filter(X_train, y_train, outlier_z, outlier_on)

    fit_scope = str(params.get("fit_scope", "train")).strip().lower()
    if fit_scope == "train+val":
        X_fit = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        y_fit = np.concatenate([y_train, y_val], axis=0)
    else:
        X_fit = X_train.reset_index(drop=True)
        y_fit = y_train

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(imputer.fit_transform(X_fit))
    X_train_scaled = scaler.transform(imputer.transform(X_train))
    X_val_scaled = scaler.transform(imputer.transform(X_val))
    X_test_scaled = scaler.transform(imputer.transform(X_test))

    C, gamma, epsilon = _resolve_svr_params(params, selection)
    model = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)
    model.fit(X_fit_scaled, y_fit)

    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_out = pd.DataFrame({"y_true": y_train, "y_pred": y_train_pred})
    val_out = pd.DataFrame({"y_true": y_val, "y_pred": y_val_pred})
    test_out = pd.DataFrame({"y_true": y_test, "y_pred": y_test_pred})

    train_out.to_csv(run_dir / "predictions_train.csv", index=False)
    val_out.to_csv(run_dir / "predictions_val.csv", index=False)
    test_out.to_csv(run_dir / "predictions_test.csv", index=False)

    metrics_payload = _load_json(metrics_path)
    metrics_payload.update(
        {
            "train": _compute_metrics(y_train, y_train_pred),
            "val": _compute_metrics(y_val, y_val_pred),
            "test": _compute_metrics(y_test, y_test_pred),
        }
    )
    _write_metrics_if_changed(metrics_path, metrics_payload)

    if mode == "train":
        return train_out, metrics_payload
    if mode == "val":
        return val_out, metrics_payload
    return test_out, metrics_payload
