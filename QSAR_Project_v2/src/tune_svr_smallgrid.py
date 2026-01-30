from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
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
DEFAULT_GAMMA = 0.0137853223
DEFAULT_EPSILON = 0.0805095587

C_MULTIPLIERS = [0.3, 0.6, 1.0, 1.5, 2.5, 3.0]
GAMMA_MULTIPLIERS = [0.3, 0.6, 1.0, 1.5, 2.5, 3.0]
EPSILON_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0]

FEATSET_TAGS = {
    "cdft": "CDFT",
    "rdkit": "RDKIT",
    "both": "BOTH",
}


@dataclass(frozen=True)
class TuneConfig:
    seed: int
    target_col: str
    refit_on_trainval: bool
    run_name: str | None
    robust_select: bool
    delta: float
    feature_set: str
    use_cv: bool = False
    cv_folds: int = 5
    cv_repeats: int = 3
    cv_shuffle: bool = True
    cv_scoring: str = "rmse"
    smooth_search: bool = False
    stage1_c_grid: List[float] = None  # type: ignore[assignment]
    stage1_gamma_grid: List[float] = None  # type: ignore[assignment]
    stage1_eps_grid: List[float] = None  # type: ignore[assignment]
    stage2_factor: float = 3.0
    stage2_points: int = 5
    stage2_eps_points: int = 5
    corr_threshold: float = 0.0
    outlier_z: float = 0.0
    outlier_on: str = "y"
    auto_plot_test: bool = False


def _parse_bool(text: str) -> bool:
    t = text.strip().lower()
    if t in {"1", "true", "t", "yes", "y"}:
        return True
    if t in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Could not parse boolean value from: {text!r}")


def _parse_float_list(text: str) -> List[float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("Expected a comma-separated list of floats.")
    return [float(p) for p in parts]


def _make_candidates_from_grid(
    c_vals: List[float], g_vals: List[float], e_vals: List[float]
) -> List[Tuple[float, float, float]]:
    return [(float(C), float(g), float(e)) for C in c_vals for g in g_vals for e in e_vals]


def _logspace_around(value: float, factor: float, points: int) -> List[float]:
    low = max(value / factor, 1e-12)
    high = value * factor
    return list(np.logspace(np.log10(low), np.log10(high), points))


def _linspace_around(value: float, factor: float, points: int) -> List[float]:
    low = max(1e-6, value / factor)
    high = value * factor
    return list(np.linspace(low, high, points))


def _split_seed_from_manifest(project_root: Path) -> str:
    manifest = load_split_manifest(project_root)
    return split_seed_from_manifest(manifest)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
    }


def _axis_limits_from_true(values: Iterable[np.ndarray], pad: float = 0.5) -> Tuple[float, float]:
    all_true = np.concatenate([np.asarray(v).ravel() for v in values])
    vmin = float(np.min(all_true) - pad)
    vmax = float(np.max(all_true) + pad)
    return vmin, vmax


def _next_plot_path(figures_dir: Path, tag: str = "SVR_TUNED_TEST") -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    pattern = f"Academic_Plot_{tag}_*.png"
    existing = sorted(figures_dir.glob(pattern))
    if not existing:
        counter = 1
    else:
        counters: List[int] = []
        for p in existing:
            stem = p.stem
            try:
                counters.append(int(stem.split("_")[-1]))
            except ValueError:
                continue
        counter = (max(counters) + 1) if counters else 1
    return figures_dir / f"Academic_Plot_{tag}_{counter}.png"


def _build_run_dir(project_root: Path, seed: int, feature_set: str, run_name: str | None) -> Path:
    date_str = datetime.now().strftime("%Y-%m-%d")
    split_seed = _split_seed_from_manifest(project_root)
    feat_tag = FEATSET_TAGS.get(feature_set, "RAW")
    base = f"{date_str}_svr_tuned_seed{seed}_split{split_seed}_feats{feat_tag}"
    if run_name:
        safe = run_name.strip().replace(" ", "_")
        base = f"{base}_{safe}"
    run_dir = project_root / "results" / "runs" / base
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return run_dir


def _academic_style() -> None:
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


def _select_params_standard(grid_rows: List[Dict[str, float]]) -> Tuple[Tuple[float, float, float], float, float]:
    best_key: Tuple[float, float] | None = None  # (val_rmse, -val_r2)
    best_params: Tuple[float, float, float] | None = None
    for row in grid_rows:
        key = (row["val_rmse"], -row["val_r2"])
        if best_key is None or key < best_key:
            best_key = key
            best_params = (row["C"], row["gamma"], row["epsilon"])
    if best_key is None or best_params is None:
        raise RuntimeError("Grid search did not produce any results.")
    best_val_rmse = float(best_key[0])
    best_val_r2 = float(-best_key[1])
    return best_params, best_val_rmse, best_val_r2


def _select_params_robust(
    grid_rows: List[Dict[str, float]], delta: float
) -> Tuple[Tuple[float, float, float], Dict[str, float], Dict[str, object]]:
    best_val_r2 = max(row["val_r2"] for row in grid_rows)
    candidates = [row for row in grid_rows if row["val_r2"] >= best_val_r2 - delta]
    if not candidates:
        raise RuntimeError("Robust selection produced no candidates.")

    def primary_key(row: Dict[str, float]) -> float:
        return abs(row["train_r2"] - row["val_r2"])

    min_gap = min(primary_key(r) for r in candidates)
    gap_candidates = [r for r in candidates if primary_key(r) == min_gap]

    decisions: List[str] = []
    if len(gap_candidates) > 1:
        decisions.append("tie on abs(train_r2 - val_r2)")

    min_val_rmse = min(r["val_rmse"] for r in gap_candidates)
    rmse_candidates = [r for r in gap_candidates if r["val_rmse"] == min_val_rmse]
    if len(rmse_candidates) < len(gap_candidates):
        decisions.append("tie-break: lower val_rmse")

    min_C = min(r["C"] for r in rmse_candidates)
    c_candidates = [r for r in rmse_candidates if r["C"] == min_C]
    if len(c_candidates) < len(rmse_candidates):
        decisions.append("tie-break: smaller C")

    min_gamma = min(r["gamma"] for r in c_candidates)
    chosen = [r for r in c_candidates if r["gamma"] == min_gamma]
    if len(chosen) < len(c_candidates):
        decisions.append("tie-break: smaller gamma")

    chosen_row = chosen[0]

    chosen_params = (chosen_row["C"], chosen_row["gamma"], chosen_row["epsilon"])
    chosen_metrics = {
        "chosen_train_r2": float(chosen_row["train_r2"]),
        "chosen_val_r2": float(chosen_row["val_r2"]),
        "chosen_val_rmse": float(chosen_row["val_rmse"]),
    }
    selection_audit: Dict[str, object] = {
        "best_val_r2": float(best_val_r2),
        "delta": float(delta),
        "candidate_count": int(len(candidates)),
        "chosen_params": {
            "C": float(chosen_row["C"]),
            "gamma": float(chosen_row["gamma"]),
            "epsilon": float(chosen_row["epsilon"]),
        },
        **chosen_metrics,
        "tie_break_decisions": decisions,
    }
    return chosen_params, chosen_metrics, selection_audit


def _evaluate_candidates_cv(
    candidates: List[Tuple[float, float, float]],
    X: pd.DataFrame,
    y: np.ndarray,
    folds: int,
    repeats: int,
    shuffle: bool,
    seed: int,
) -> List[Dict[str, float]]:
    if not shuffle:
        cv = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=None)
    else:
        cv = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    results: List[Dict[str, float]] = []

    for C, gamma, epsilon in candidates:
        rmses: List[float] = []
        r2s: List[float] = []
        for train_idx, val_idx in cv.split(X, y):
            X_tr = X.iloc[train_idx]
            y_tr = y[train_idx]
            X_va = X.iloc[val_idx]
            y_va = y[val_idx]

            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(imputer.fit_transform(X_tr))
            X_va_scaled = scaler.transform(imputer.transform(X_va))

            model = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)
            model.fit(X_tr_scaled, y_tr)
            y_pred = model.predict(X_va_scaled)

            rmses.append(float(np.sqrt(mean_squared_error(y_va, y_pred))))
            r2s.append(float(r2_score(y_va, y_pred)))

        results.append(
            {
                "C": float(C),
                "gamma": float(gamma),
                "epsilon": float(epsilon),
                "cv_mean_rmse": float(np.mean(rmses)),
                "cv_std_rmse": float(np.std(rmses, ddof=0)),
                "cv_mean_r2": float(np.mean(r2s)),
                "cv_std_r2": float(np.std(r2s, ddof=0)),
            }
        )

    return results


def _select_best_cv(results: List[Dict[str, float]]) -> Dict[str, float]:
    def key(row: Dict[str, float]) -> Tuple[float, float, float]:
        return (row["cv_mean_rmse"], row["cv_std_rmse"], -row["cv_mean_r2"])

    return min(results, key=key)


def _prune_correlated_features(
    X_train: pd.DataFrame, threshold: float
) -> Tuple[pd.DataFrame, List[str]]:
    corr = X_train.corr().abs()
    kept: List[str] = []
    dropped: List[str] = []
    for col in X_train.columns:
        if not kept:
            kept.append(col)
            continue
        high = corr.loc[col, kept].max()
        if float(high) > threshold:
            dropped.append(col)
        else:
            kept.append(col)
    return X_train[kept], dropped


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Small-grid SVR tuning on fixed train/val/test splits. "
            "Example: python src/tune_svr_smallgrid.py --auto-plot-test true"
        )
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed label for run naming.")
    parser.add_argument(
        "--target-col",
        type=str,
        default="log(1o2)",
        help="Target column name (required, normalized to lowercase).",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=["cdft", "rdkit", "both"],
        default="both",
        help="Feature subset to use: cdft, rdkit, or both.",
    )
    parser.add_argument(
        "--refit-on-trainval",
        type=str,
        default="true",
        help="Whether to refit best params on train+val (true/false).",
    )
    parser.add_argument(
        "--robust-select",
        type=str,
        default="false",
        help="Use robust selection within best_val_r2 - delta (true/false).",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.02,
        help="Robust selection tolerance on val_r2.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.0,
        help="Correlation threshold for pruning (0 disables).",
    )
    parser.add_argument(
        "--outlier-z",
        type=float,
        default=0.0,
        help="Z-score threshold for outlier removal (0 disables).",
    )
    parser.add_argument(
        "--outlier-on",
        type=str,
        choices=["y", "x"],
        default="y",
        help="Outlier detection on y or x.",
    )
    parser.add_argument(
        "--auto-plot-test",
        type=str,
        default="false",
        help="Generate test-only parity plot (true/false). Example: --auto-plot-test true",
    )
    parser.add_argument(
        "--use-cv",
        type=str,
        default="false",
        help="Use RepeatedKFold CV for tuning (true/false).",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--cv-repeats", type=int, default=3, help="Number of CV repeats.")
    parser.add_argument(
        "--cv-shuffle",
        type=str,
        default="true",
        help="Shuffle CV splits (true/false).",
    )
    parser.add_argument(
        "--smooth-search",
        type=str,
        default="false",
        help="Enable two-stage log-space refinement (true/false).",
    )
    parser.add_argument(
        "--stage2-factor",
        type=float,
        default=3.0,
        help="Stage2 range factor.",
    )
    parser.add_argument(
        "--stage2-points",
        type=int,
        default=5,
        help="Stage2 logspace points for C/gamma.",
    )
    parser.add_argument(
        "--stage2-eps-points",
        type=int,
        default=5,
        help="Stage2 linspace points for epsilon.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional suffix to append to the run directory name.",
    )
    args = parser.parse_args()

    cfg = TuneConfig(
        seed=int(args.seed),
        target_col=str(args.target_col),
        refit_on_trainval=_parse_bool(str(args.refit_on_trainval)),
        run_name=args.run_name,
        robust_select=_parse_bool(str(args.robust_select)),
        delta=float(args.delta),
        feature_set=str(args.feature_set),
        use_cv=_parse_bool(str(args.use_cv)),
        cv_folds=int(args.cv_folds),
        cv_repeats=int(args.cv_repeats),
        cv_shuffle=_parse_bool(str(args.cv_shuffle)),
        smooth_search=_parse_bool(str(args.smooth_search)),
        stage1_c_grid=[DEFAULT_C * m for m in C_MULTIPLIERS],
        stage1_gamma_grid=[DEFAULT_GAMMA * m for m in GAMMA_MULTIPLIERS],
        stage1_eps_grid=[DEFAULT_EPSILON * m for m in EPSILON_MULTIPLIERS],
        stage2_factor=float(args.stage2_factor),
        stage2_points=int(args.stage2_points),
        stage2_eps_points=int(args.stage2_eps_points),
        corr_threshold=float(args.corr_threshold),
        outlier_z=float(args.outlier_z),
        outlier_on=str(args.outlier_on),
        auto_plot_test=_parse_bool(str(args.auto_plot_test)),
    )

    if cfg.use_cv and cfg.robust_select:
        print("Robust-select ignored because --use-cv true.")

    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"

    train_path = processed_dir / "train_data.csv"
    val_path = processed_dir / "val_data.csv"
    test_path = processed_dir / "test_data.csv"

    train_df = replace_inf_with_nan(normalize_columns(pd.read_csv(train_path)))
    val_df = replace_inf_with_nan(normalize_columns(pd.read_csv(val_path)))
    test_df = replace_inf_with_nan(normalize_columns(pd.read_csv(test_path)))

    target_col = resolve_target_col(train_df, cfg.target_col)
    feature_cols = select_feature_columns(train_df, target_col, cfg.feature_set)
    if not feature_cols:
        raise ValueError("No numeric feature columns found after exclusions.")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].to_numpy()
    X_val = val_df[feature_cols]
    y_val = val_df[target_col].to_numpy()
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].to_numpy()

    dropped_corr: List[str] = []
    if cfg.corr_threshold and cfg.corr_threshold > 0:
        X_train, dropped_corr = _prune_correlated_features(X_train, cfg.corr_threshold)
        feature_cols = list(X_train.columns)
        X_val = X_val[feature_cols]
        X_test = X_test[feature_cols]

    removed_train_rows = 0
    if cfg.outlier_z and cfg.outlier_z > 0:
        X_train, y_train, removed_train_rows = _apply_outlier_filter(
            X_train, y_train, cfg.outlier_z, cfg.outlier_on
        )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
    X_val_scaled = scaler.transform(imputer.transform(X_val))
    X_test_scaled = scaler.transform(imputer.transform(X_test))

    stage1_candidates = _make_candidates_from_grid(
        cfg.stage1_c_grid, cfg.stage1_gamma_grid, cfg.stage1_eps_grid
    )

    grid_rows: List[Dict[str, float]] = []
    selection_rows: List[Dict[str, float]] = []
    cv_rows: List[Dict[str, float]] = []

    total = len(stage1_candidates)
    if cfg.smooth_search:
        total += cfg.stage2_points * cfg.stage2_points * cfg.stage2_eps_points
    print(f"Grid size: {total} combinations")

    if cfg.use_cv:
        if cfg.refit_on_trainval:
            X_pool = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
            y_pool = np.concatenate([y_train, y_val], axis=0)
        else:
            X_pool = X_train.reset_index(drop=True)
            y_pool = y_train

        cv_stage1 = _evaluate_candidates_cv(
            stage1_candidates,
            X_pool,
            y_pool,
            cfg.cv_folds,
            cfg.cv_repeats,
            cfg.cv_shuffle,
            cfg.seed,
        )
        cv_map = {(r["C"], r["gamma"], r["epsilon"]): r for r in cv_stage1}
    else:
        cv_map = {}

    def add_candidates(candidates: List[Tuple[float, float, float]], stage: int) -> None:
        nonlocal grid_rows, selection_rows, cv_rows
        for C, gamma, epsilon in candidates:
            model = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)
            model.fit(X_train_scaled, y_train)

            train_pred = model.predict(X_train_scaled)
            val_pred = model.predict(X_val_scaled)

            train_metrics = _compute_metrics(y_train, train_pred)
            val_metrics = _compute_metrics(y_val, val_pred)

            row = {
                "stage": stage,
                "C": float(C),
                "gamma": float(gamma),
                "epsilon": float(epsilon),
                "val_r2": float(val_metrics["r2"]),
                "val_mae": float(val_metrics["mae"]),
                "val_rmse": float(val_metrics["rmse"]),
            }
            if cfg.use_cv:
                cv_row = cv_map.get((float(C), float(gamma), float(epsilon)))
                if cv_row:
                    row.update(
                        {
                            "cv_mean_rmse": float(cv_row["cv_mean_rmse"]),
                            "cv_std_rmse": float(cv_row["cv_std_rmse"]),
                            "cv_mean_r2": float(cv_row["cv_mean_r2"]),
                            "cv_std_r2": float(cv_row["cv_std_r2"]),
                        }
                    )
                    cv_rows.append(cv_row)

            grid_rows.append(row)
            selection_rows.append(
                {
                    "stage": stage,
                    "C": float(C),
                    "gamma": float(gamma),
                    "epsilon": float(epsilon),
                    "train_r2": float(train_metrics["r2"]),
                    "val_r2": float(val_metrics["r2"]),
                    "val_rmse": float(val_metrics["rmse"]),
                }
            )

    add_candidates(stage1_candidates, stage=1)

    if cfg.smooth_search:
        if cfg.use_cv:
            best_stage1 = _select_best_cv(list(cv_map.values()))
            best_C1 = float(best_stage1["C"])
            best_G1 = float(best_stage1["gamma"])
            best_E1 = float(best_stage1["epsilon"])
        else:
            if cfg.robust_select:
                stage1_sel = [r for r in selection_rows if r["stage"] == 1]
                best_stage1_params, _, _ = _select_params_robust(stage1_sel, cfg.delta)
            else:
                best_stage1_params, _, _ = _select_params_standard(
                    [r for r in grid_rows if r["stage"] == 1]
                )
            best_C1, best_G1, best_E1 = best_stage1_params

        C_local = _logspace_around(best_C1, cfg.stage2_factor, cfg.stage2_points)
        G_local = _logspace_around(best_G1, cfg.stage2_factor, cfg.stage2_points)
        E_local = _linspace_around(best_E1, cfg.stage2_factor, cfg.stage2_eps_points)
        stage2_candidates = _make_candidates_from_grid(C_local, G_local, E_local)

        if cfg.use_cv:
            cv_stage2 = _evaluate_candidates_cv(
                stage2_candidates,
                X_pool,
                y_pool,
                cfg.cv_folds,
                cfg.cv_repeats,
                cfg.cv_shuffle,
                cfg.seed,
            )
            for r in cv_stage2:
                cv_map[(r["C"], r["gamma"], r["epsilon"])] = r

        add_candidates(stage2_candidates, stage=2)

    if cfg.use_cv:
        all_cv = [cv_map[k] for k in cv_map]
        if cfg.robust_select:
            best_rmse = min(r["cv_mean_rmse"] for r in all_cv)
            threshold = best_rmse + cfg.delta
            candidates = [r for r in all_cv if r["cv_mean_rmse"] <= threshold]
            candidates = sorted(
                candidates,
                key=lambda r: (r["cv_std_rmse"], r["C"], r["gamma"]),
            )
            best_cv = candidates[0]
            selection_audit = {
                "mode": "cv_robust",
                "delta": cfg.delta,
                "candidate_count": len(candidates),
                "tie_breaks": ["lower cv_std_rmse", "smaller C", "smaller gamma"],
            }
        else:
            best_cv = min(all_cv, key=lambda r: (r["cv_mean_rmse"], -r["cv_mean_r2"]))
            selection_audit = {"mode": "cv"}

        best_C = float(best_cv["C"])
        best_gamma = float(best_cv["gamma"])
        best_epsilon = float(best_cv["epsilon"])
        best_val_rmse = float(best_cv["cv_mean_rmse"])
        best_val_r2 = float(best_cv["cv_mean_r2"])
        chosen_metrics = {}
        selection_audit.update(
            {
                "chosen_params": {
                    "C": float(best_C),
                    "gamma": float(best_gamma),
                    "epsilon": float(best_epsilon),
                },
                "chosen_cv_mean_rmse": float(best_val_rmse),
                "chosen_cv_mean_r2": float(best_val_r2),
            }
        )
    else:
        if cfg.robust_select:
            best_params, chosen_metrics, selection_audit = _select_params_robust(
                selection_rows, cfg.delta
            )
            best_C, best_gamma, best_epsilon = best_params
            best_val_rmse = min(row["val_rmse"] for row in selection_rows)
            best_val_r2 = max(row["val_r2"] for row in selection_rows)
            selection_audit["mode"] = "val_robust"
        else:
            best_params, best_val_rmse, best_val_r2 = _select_params_standard(grid_rows)
            best_C, best_gamma, best_epsilon = best_params
            chosen_metrics = {}
            selection_audit = {"mode": "val"}

    if cfg.refit_on_trainval:
        X_refit = pd.concat([X_train, X_val], axis=0)
        y_refit = np.concatenate([y_train, y_val], axis=0)

        refit_imputer = SimpleImputer(strategy="median")
        refit_scaler = StandardScaler()

        X_refit_scaled = refit_scaler.fit_transform(refit_imputer.fit_transform(X_refit))
        X_train_eval_scaled = refit_scaler.transform(refit_imputer.transform(X_train))
        X_val_eval_scaled = refit_scaler.transform(refit_imputer.transform(X_val))
        X_test_eval_scaled = refit_scaler.transform(refit_imputer.transform(X_test))

        final_model = SVR(kernel="rbf", C=best_C, gamma=best_gamma, epsilon=best_epsilon)
        final_model.fit(X_refit_scaled, y_refit)

        y_train_pred = final_model.predict(X_train_eval_scaled)
        y_val_pred = final_model.predict(X_val_eval_scaled)
        y_test_pred = final_model.predict(X_test_eval_scaled)

        final_imputer = refit_imputer
        final_scaler = refit_scaler
        fit_scope = "train+val"
    else:
        final_model = SVR(kernel="rbf", C=best_C, gamma=best_gamma, epsilon=best_epsilon)
        final_model.fit(X_train_scaled, y_train)

        y_train_pred = final_model.predict(X_train_scaled)
        y_val_pred = final_model.predict(X_val_scaled)
        y_test_pred = final_model.predict(X_test_scaled)

        final_imputer = imputer
        final_scaler = scaler
        fit_scope = "train"

    train_metrics = _compute_metrics(y_train, y_train_pred)
    val_metrics = _compute_metrics(y_val, y_val_pred)
    test_metrics = _compute_metrics(y_test, y_test_pred)

    if "chosen_params" not in selection_audit:
        selection_audit["chosen_params"] = {
            "C": float(best_C),
            "gamma": float(best_gamma),
            "epsilon": float(best_epsilon),
        }
    if "chosen_val_rmse" not in selection_audit:
        selection_audit["chosen_val_rmse"] = float(best_val_rmse)
    if "chosen_val_r2" not in selection_audit:
        selection_audit["chosen_val_r2"] = float(best_val_r2)

    run_dir = _build_run_dir(project_root, cfg.seed, cfg.feature_set, cfg.run_name)

    params_path = run_dir / "params.json"
    metrics_path = run_dir / "metrics.json"
    grid_path = run_dir / "grid_results.csv"
    pred_val_path = run_dir / "predictions_val.csv"
    pred_test_path = run_dir / "predictions_test.csv"
    model_path = run_dir / "model.pkl"
    selection_path = run_dir / "selection.json"

    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(grid_path, index=False)

    val_out = pd.DataFrame({"y_true": y_val, "y_pred": y_val_pred})
    test_out = pd.DataFrame({"y_true": y_test, "y_pred": y_test_pred})
    val_out.to_csv(pred_val_path, index=False)
    test_out.to_csv(pred_test_path, index=False)

    split_seed = _split_seed_from_manifest(project_root)

    params_payload = {
        "seed": cfg.seed,
        "target_col": target_col,
        "fit_scope": fit_scope,
        "feature_set": cfg.feature_set,
        "split_seed": split_seed,
        "n_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "corr_threshold": float(cfg.corr_threshold),
        "dropped_correlated_features": dropped_corr,
        "outlier_z": float(cfg.outlier_z),
        "outlier_on": cfg.outlier_on,
        "removed_train_rows_count": int(removed_train_rows),
        "best_params": {
            "C": float(best_C),
            "gamma": float(best_gamma),
            "epsilon": float(best_epsilon),
        },
        "defaults": {
            "C": DEFAULT_C,
            "gamma": DEFAULT_GAMMA,
            "epsilon": DEFAULT_EPSILON,
        },
        "grid_definition": {
            "C_multipliers": C_MULTIPLIERS,
            "gamma_multipliers": GAMMA_MULTIPLIERS,
            "epsilon_multipliers": EPSILON_MULTIPLIERS,
            "grid_size": total,
        },
        "cv": {
            "enabled": cfg.use_cv,
            "folds": cfg.cv_folds,
            "repeats": cfg.cv_repeats,
            "shuffle": cfg.cv_shuffle,
        },
        "smooth_search": {
            "enabled": cfg.smooth_search,
            "stage2_factor": cfg.stage2_factor,
            "stage2_points": cfg.stage2_points,
            "stage2_eps_points": cfg.stage2_eps_points,
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    params_path.write_text(json.dumps(params_payload, indent=2), encoding="utf-8")

    metrics_payload = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "best_val_rmse": float(best_val_rmse),
        "best_val_r2": float(best_val_r2),
    }
    if cfg.use_cv:
        best_cv = min(
            cv_rows,
            key=lambda r: (r["cv_mean_rmse"], -r["cv_mean_r2"]),
        ) if cv_rows else None
        metrics_payload["cv"] = {
            "folds": cfg.cv_folds,
            "repeats": cfg.cv_repeats,
            "mean_rmse": float(best_cv["cv_mean_rmse"]) if best_cv else None,
            "std_rmse": float(best_cv["cv_std_rmse"]) if best_cv else None,
            "mean_r2": float(best_cv["cv_mean_r2"]) if best_cv else None,
            "std_r2": float(best_cv["cv_std_r2"]) if best_cv else None,
            "best_params": {
                "C": float(best_C),
                "gamma": float(best_gamma),
                "epsilon": float(best_epsilon),
            },
        }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    selection_payload = dict(selection_audit)
    selection_payload.update(chosen_metrics)
    selection_path.write_text(json.dumps(selection_payload, indent=2), encoding="utf-8")

    with model_path.open("wb") as f:
        pickle.dump(
            {
                "imputer": final_imputer,
                "scaler": final_scaler,
                "model": final_model,
                "feature_columns": feature_cols,
                "target_column": target_col,
                "fit_scope": fit_scope,
                "feature_set": cfg.feature_set,
                "split_seed": split_seed,
            },
            f,
        )

    if cfg.auto_plot_test:
        _academic_style()
        fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
        lim_min, lim_max = _axis_limits_from_true([y_test], pad=0.5)

        ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", color="#555555", lw=1.5)
        ax.scatter(
            y_test,
            y_test_pred,
            s=80,
            alpha=0.85,
            color="#C44E52",
            edgecolors="#800000",
            linewidths=0.8,
            label=f"Test (R2={test_metrics['r2']:.3f}, RMSE={test_metrics['rmse']:.3f})",
        )

        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Experimental log(1/O2)")
        ax.set_ylabel("Predicted log(1/O2)")
        ax.set_title("Experimental vs. Predicted Values")
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.35)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.6)
        ax.legend(frameon=False, loc="best")

        plot_path = _next_plot_path(run_dir / "figures", tag="SVR_TUNED_TEST")
        fig.savefig(plot_path, dpi=600, bbox_inches="tight")

    print("Best params:", {"C": best_C, "gamma": best_gamma, "epsilon": best_epsilon})
    print("CV used:", cfg.use_cv)
    print("Corr pruning enabled:", bool(cfg.corr_threshold and cfg.corr_threshold > 0))
    print("Outlier handling enabled:", bool(cfg.outlier_z and cfg.outlier_z > 0))
    print("Metrics summary:", {"train": train_metrics, "val": val_metrics, "test": test_metrics})
    print("Run directory:")
    print(run_dir)

    cmd = (
        f"python src\\tune_svr_smallgrid.py --seed {cfg.seed} "
        f"--target-col \"{target_col}\" --feature-set {cfg.feature_set} "
        f"--refit-on-trainval {str(cfg.refit_on_trainval).lower()} "
        f"--robust-select {str(cfg.robust_select).lower()} --delta {cfg.delta} "
        f"--use-cv {str(cfg.use_cv).lower()} --cv-folds {cfg.cv_folds} "
        f"--cv-repeats {cfg.cv_repeats} --cv-shuffle {str(cfg.cv_shuffle).lower()} "
        f"--smooth-search {str(cfg.smooth_search).lower()} --stage2-factor {cfg.stage2_factor} "
        f"--stage2-points {cfg.stage2_points} --stage2-eps-points {cfg.stage2_eps_points} "
        f"--corr-threshold {cfg.corr_threshold} --outlier-z {cfg.outlier_z} "
        f"--outlier-on {cfg.outlier_on} --auto-plot-test {str(cfg.auto_plot_test).lower()}"
    )
    if cfg.run_name:
        cmd += f" --run-name \"{cfg.run_name}\""
    print("Reproduce command:")
    print(cmd)


if __name__ == "__main__":
    main()
