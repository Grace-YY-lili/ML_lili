from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from run_artifacts import (
    load_or_write_predictions_and_metrics,
    load_split_manifest,
    resolve_run_dir,
    split_seed_from_manifest,
)

R2_TOL = 1e-9


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


def _axis_limits_from_true(values: Iterable[np.ndarray], pad: float = 0.5) -> Tuple[float, float]:
    all_true = np.concatenate([np.asarray(v).ravel() for v in values if v.size])
    vmin = float(np.min(all_true) - pad)
    vmax = float(np.max(all_true) + pad)
    return vmin, vmax


def _safe_tag(tag: str | None) -> str:
    if not tag:
        return ""
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in tag.strip())
    return safe.upper()


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_split_seed(run_dir: Path, project_root: Path) -> str:
    manifest = load_split_manifest(project_root)
    manifest_seed = split_seed_from_manifest(manifest)
    params_path = run_dir / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing params.json in run dir: {run_dir}")
    params = _load_json(params_path)
    params_seed = params.get("split_seed")
    if params_seed is None:
        raise ValueError("params.json missing split_seed; cannot validate split manifest.")
    if str(params_seed) != str(manifest_seed):
        raise ValueError(
            f"Split seed mismatch: run expects {params_seed}, manifest has {manifest_seed}."
        )
    return str(params_seed)


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")


def _verify_test_r2(pred_df: pd.DataFrame, metrics: Dict[str, object]) -> Tuple[float, float]:
    y_true_col, y_pred_col = _get_pred_cols(pred_df)
    y_true = pred_df[y_true_col].to_numpy()
    y_pred = pred_df[y_pred_col].to_numpy()
    r2_calc = float(r2_score(y_true, y_pred))
    test_metrics = metrics.get("test") if isinstance(metrics.get("test"), dict) else {}
    r2_reported = None
    if isinstance(test_metrics, dict) and "r2" in test_metrics:
        r2_reported = float(test_metrics["r2"])
        if abs(r2_calc - r2_reported) > R2_TOL:
            raise ValueError(
                f"metrics.json test.r2 {r2_reported} != predictions_test r2 {r2_calc}."
            )
    else:
        raise ValueError("metrics.json missing test.r2; cannot validate test parity.")
    return r2_calc, r2_reported


def _load_predictions(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    pred_df = pd.read_csv(path)
    y_true_col, y_pred_col = _get_pred_cols(pred_df)
    return pred_df[y_true_col].to_numpy(), pred_df[y_pred_col].to_numpy()


def _read_params(run_dir: Path) -> Dict[str, object]:
    params_path = run_dir / "params.json"
    return _load_json(params_path) if params_path.exists() else {}


def _feature_tag_from_leaf(leaf: str) -> str:
    lower = leaf.lower()
    if "featscdft" in lower:
        return "CDFT"
    if "featsrdkit" in lower:
        return "RDKIT"
    if "featsboth" in lower:
        return "BOTH"
    return ""


def _feature_tag_from_params(params: Dict[str, object]) -> str:
    value = str(params.get("feature_set", "")).lower()
    if value == "cdft":
        return "CDFT"
    if value == "rdkit":
        return "RDKIT"
    if value == "both":
        return "BOTH"
    return ""


def _run_name_suffix(leaf: str) -> str:
    lower = leaf.lower()
    idx = lower.find("feats")
    if idx == -1:
        return ""
    rest = leaf[idx + len("feats") :]
    if "_" not in rest:
        return ""
    _, suffix = rest.split("_", 1)
    return suffix


def _find_matching_run(
    project_root: Path,
    base_run_dir: Path,
    target_tag: str,
) -> Path:
    base_leaf = base_run_dir.name
    base_tag = _feature_tag_from_leaf(base_leaf)
    if base_tag:
        replaced = base_leaf.replace(f"feats{base_tag}", f"feats{target_tag}")
        candidate = resolve_run_dir(project_root, replaced)
        if candidate.exists():
            return candidate

    base_params = _read_params(base_run_dir)
    base_seed = str(base_params.get("seed", ""))
    base_split = str(base_params.get("split_seed", ""))
    base_suffix = _run_name_suffix(base_leaf)

    search_roots = [
        project_root / "results" / "runs",
        project_root / "results" / "archive",
    ]
    matches: List[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob(f"*feats{target_tag}*"):
            if not p.is_dir():
                continue
            params = _read_params(p)
            split_seed = str(params.get("split_seed", ""))
            seed = str(params.get("seed", ""))
            suffix = _run_name_suffix(p.name)
            if base_split and split_seed and split_seed != base_split:
                continue
            if base_seed and seed and seed != base_seed:
                continue
            if base_suffix and suffix and suffix != base_suffix:
                continue
            matches.append(p)

    if not matches:
        raise FileNotFoundError(
            f"Missing run for feature-set {target_tag}. "
            f"Tried replacing feats{base_tag or '???'} in {base_leaf} and global search."
        )
    return max(matches, key=lambda d: d.stat().st_mtime)


def _plot_parity(
    run_dir: Path,
    out_dir: Path,
    tag: str,
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_val: np.ndarray,
    y_val_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    test_r2_label: float,
    test_rmse_label: float,
) -> Tuple[Path, str]:
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

    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    lim_min, lim_max = _axis_limits_from_true([y_train, y_val, y_test], pad=0.5)

    ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", color="#555555", lw=1.5)

    if y_train.size:
        r2_train = float(r2_score(y_train, y_train_pred))
        ax.scatter(
            y_train,
            y_train_pred,
            s=50,
            alpha=0.25,
            color="#4682B4",
            edgecolors="none",
            label=f"Train (R2={r2_train:.3f})",
        )

    if y_val.size:
        r2_val = float(r2_score(y_val, y_val_pred))
        ax.scatter(
            y_val,
            y_val_pred,
            s=60,
            alpha=0.6,
            color="#55A868",
            marker="^",
            edgecolors="none",
            label=f"Val (R2={r2_val:.3f})",
        )

    test_label = f"Test (R2={test_r2_label:.3f}, RMSE={test_rmse_label:.3f})"
    ax.scatter(
        y_test,
        y_test_pred,
        s=80,
        alpha=0.85,
        color="#C44E52",
        edgecolors="#800000",
        linewidths=0.8,
        label=test_label,
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

    run_leaf = Path(run_dir).name
    safe_tag = _safe_tag(tag)
    if not safe_tag:
        raise ValueError("tag must be provided for standardized output naming.")
    stem = f"{run_leaf}__{safe_tag}__PARITY"
    out_path = out_dir / f"{stem}.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")

    return out_path, test_label


def plot_from_run(
    run_dir_arg: str,
    out_dir_arg: str,
    tag: str | None = None,
    target_col: str = "log(1o2)",
    feature_set: str = "both",
) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    run_dir = resolve_run_dir(project_root, run_dir_arg)

    _verify_split_seed(run_dir, project_root)

    metrics_path = run_dir / "metrics.json"
    pred_test_path = run_dir / "predictions_test.csv"
    _require_file(metrics_path, "metrics.json")
    if not pred_test_path.exists():
        pred_test_df, metrics_payload = load_or_write_predictions_and_metrics(
            run_dir,
            mode="test",
            project_root=project_root,
            target_col=target_col,
            feature_set=feature_set,
        )
    else:
        metrics_payload = _load_json(metrics_path)
        pred_test_df = pd.read_csv(pred_test_path)
    test_r2_calc, test_r2_reported = _verify_test_r2(pred_test_df, metrics_payload)
    y_true_col, y_pred_col = _get_pred_cols(pred_test_df)
    y_test = pred_test_df[y_true_col].to_numpy()
    y_test_pred = pred_test_df[y_pred_col].to_numpy()
    test_metrics = metrics_payload.get("test") if isinstance(metrics_payload.get("test"), dict) else {}
    test_r2_label = float(test_metrics.get("r2", test_r2_reported))
    test_rmse_label = float(test_metrics.get("rmse", np.sqrt(mean_squared_error(y_test, y_test_pred))))

    pred_val_path = run_dir / "predictions_val.csv"
    y_val = np.array([])
    y_val_pred = np.array([])
    if pred_val_path.exists():
        y_val, y_val_pred = _load_predictions(pred_val_path)
    else:
        pred_val_df, metrics_payload = load_or_write_predictions_and_metrics(
            run_dir,
            mode="val",
            project_root=project_root,
            target_col=target_col,
            feature_set=feature_set,
        )
        yv_true_col, yv_pred_col = _get_pred_cols(pred_val_df)
        y_val = pred_val_df[yv_true_col].to_numpy()
        y_val_pred = pred_val_df[yv_pred_col].to_numpy()

    pred_train_path = run_dir / "predictions_train.csv"
    y_train = np.array([])
    y_train_pred = np.array([])
    if pred_train_path.exists():
        y_train, y_train_pred = _load_predictions(pred_train_path)
    else:
        pred_train_df, metrics_payload = load_or_write_predictions_and_metrics(
            run_dir,
            mode="train",
            project_root=project_root,
            target_col=target_col,
            feature_set=feature_set,
        )
        yt_true_col, yt_pred_col = _get_pred_cols(pred_train_df)
        y_train = pred_train_df[yt_true_col].to_numpy()
        y_train_pred = pred_train_df[yt_pred_col].to_numpy()

    standard_out_dir = (project_root / "results" / "figures").resolve()
    requested_out = Path(out_dir_arg)
    if requested_out.is_absolute():
        requested_out = requested_out.resolve()
    else:
        requested_out = (project_root / requested_out).resolve()
    if requested_out != standard_out_dir:
        raise ValueError(
            f"Outputs must be written to {standard_out_dir}. "
            f"Received out_dir={out_dir_arg!r}."
        )
    standard_out_dir.mkdir(parents=True, exist_ok=True)

    safe_tag = _safe_tag(tag)
    out_path, test_label = _plot_parity(
        run_dir=run_dir,
        out_dir=standard_out_dir,
        tag=safe_tag,
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_val=y_val,
        y_val_pred=y_val_pred,
        y_test=y_test,
        y_test_pred=y_test_pred,
        test_r2_label=test_r2_label,
        test_rmse_label=test_rmse_label,
    )

    print(
        "Verified test.r2:",
        {"metrics.json": f"{test_r2_reported:.12f}", "recomputed": f"{test_r2_calc:.12f}"},
    )
    print(f"Parity legend uses: {test_label}")
    return out_path


def plot_from_run_multi(
    run_dir_arg: str,
    out_dir_arg: str,
    tags: List[str],
    target_col: str = "log(1o2)",
    feature_set: str = "both",
) -> List[Path]:
    out_paths: List[Path] = []
    project_root = Path(__file__).resolve().parents[1]
    base_run = resolve_run_dir(project_root, run_dir_arg)
    base_tag = _feature_tag_from_leaf(base_run.name)
    if not base_tag:
        params = _read_params(base_run)
        base_tag = _feature_tag_from_params(params)
    tag_to_run: Dict[str, Path] = {base_tag or "BOTH": base_run}

    for tag in tags:
        run_dir = tag_to_run.get(tag)
        if run_dir is None:
            run_dir = _find_matching_run(project_root, base_run, tag)
            tag_to_run[tag] = run_dir
        print(f"[{tag}] run-dir: {run_dir}")
        out_paths.append(
            plot_from_run(
                run_dir_arg=str(run_dir),
                out_dir_arg=out_dir_arg,
                tag=tag,
                target_col=target_col,
                feature_set=feature_set,
            )
        )
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified parity plotter for a run directory.")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory containing predictions/metrics artifacts.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/figures",
        help="Root output directory for parity figures.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="BOTH",
        help="Tag appended to the output filename (default: BOTH).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate BOTH/CDFT/RDKIT parity plots from the same run.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="log(1o2)",
        help="Target column name (normalized to lowercase).",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="both",
        choices=["cdft", "rdkit", "both"],
        help="Feature set for regen if predictions are missing.",
    )
    args = parser.parse_args()

    if args.all:
        out_paths = plot_from_run_multi(
            run_dir_arg=args.run_dir,
            out_dir_arg=args.out_dir,
            tags=["BOTH", "CDFT", "RDKIT"],
            target_col=args.target_col,
            feature_set=args.feature_set,
        )
        for out_path in out_paths:
            print(out_path)
    else:
        out_path = plot_from_run(
            run_dir_arg=args.run_dir,
            out_dir_arg=args.out_dir,
            tag=args.tag,
            target_col=args.target_col,
            feature_set=args.feature_set,
        )
        print(out_path)


if __name__ == "__main__":
    main()
