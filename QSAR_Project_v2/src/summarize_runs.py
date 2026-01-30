from __future__ import annotations

import argparse
import csv
import fnmatch
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RunRecord:
    run_name: str
    run_dir: str
    run_dir_leaf: str
    run_dir_relative: str
    metric_key: str
    metric_value: float
    params_json: str
    timestamp_utc: str


def _get_dot_path(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(part)
        cur = cur[part]
    return cur


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


def _metric_xlabel(metric_key: str) -> str:
    parts = metric_key.split(".")
    if len(parts) >= 2:
        split = parts[0]
        metric = parts[-1]
        return f"{split} {metric}"
    return metric_key


def _matches_any(name: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def _should_include(name: str, includes: List[str], excludes: List[str]) -> bool:
    if includes and not _matches_any(name, includes):
        return False
    if excludes and _matches_any(name, excludes):
        return False
    return True


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _params_json_string(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        payload = _load_json(path)
        return json.dumps(payload, sort_keys=True)
    except Exception:
        return ""


def _timestamp_from_mtime(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except Exception:
        return ""


def _collect_records(
    runs_dir: Path,
    metric_key: str,
    include_globs: List[str],
    exclude_globs: List[str],
) -> Tuple[List[RunRecord], Dict[str, int]]:
    records: List[RunRecord] = []
    skipped = {
        "excluded_by_filter": 0,
        "missing_metrics": 0,
        "invalid_metrics": 0,
        "missing_metric_key": 0,
        "invalid_metric_value": 0,
    }

    for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
        run_name = run_dir.name
        if not _should_include(run_name, include_globs, exclude_globs):
            skipped["excluded_by_filter"] += 1
            continue

        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            skipped["missing_metrics"] += 1
            continue

        try:
            metrics_payload = _load_json(metrics_path)
        except Exception:
            skipped["invalid_metrics"] += 1
            continue

        try:
            metric_value = _get_dot_path(metrics_payload, metric_key)
        except Exception:
            skipped["missing_metric_key"] += 1
            continue

        try:
            metric_value_f = float(metric_value)
        except Exception:
            skipped["invalid_metric_value"] += 1
            continue

        params_path = run_dir / "params.json"
        run_dir_leaf = run_dir.name
        run_dir_relative = str(Path("results") / "runs" / run_dir_leaf)
        record = RunRecord(
            run_name=run_name,
            run_dir=str(run_dir),
            run_dir_leaf=run_dir_leaf,
            run_dir_relative=run_dir_relative,
            metric_key=metric_key,
            metric_value=metric_value_f,
            params_json=_params_json_string(params_path),
            timestamp_utc=_timestamp_from_mtime(metrics_path),
        )
        records.append(record)

    return records, skipped


def _collect_records_from_paths(
    run_dirs: List[Path],
    metric_key: str,
    project_root: Path,
) -> Tuple[List[RunRecord], Dict[str, int]]:
    records: List[RunRecord] = []
    skipped = {
        "missing_dirs": 0,
        "missing_metrics": 0,
        "invalid_metrics": 0,
        "missing_metric_key": 0,
        "invalid_metric_value": 0,
    }

    for run_dir in run_dirs:
        if not run_dir.exists() or not run_dir.is_dir():
            skipped["missing_dirs"] += 1
            continue

        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            skipped["missing_metrics"] += 1
            continue

        try:
            metrics_payload = _load_json(metrics_path)
        except Exception:
            skipped["invalid_metrics"] += 1
            continue

        try:
            metric_value = _get_dot_path(metrics_payload, metric_key)
        except Exception:
            skipped["missing_metric_key"] += 1
            continue

        try:
            metric_value_f = float(metric_value)
        except Exception:
            skipped["invalid_metric_value"] += 1
            continue

        params_path = run_dir / "params.json"
        run_dir_leaf = run_dir.name
        run_dir_relative = ""
        try:
            run_dir_relative = str(run_dir.relative_to(project_root))
        except Exception:
            run_dir_relative = str(Path("results") / "runs" / run_dir_leaf)

        record = RunRecord(
            run_name=run_dir_leaf,
            run_dir=str(run_dir),
            run_dir_leaf=run_dir_leaf,
            run_dir_relative=run_dir_relative,
            metric_key=metric_key,
            metric_value=metric_value_f,
            params_json=_params_json_string(params_path),
            timestamp_utc=_timestamp_from_mtime(metrics_path),
        )
        records.append(record)

    return records, skipped


def _load_manifest_paths(manifest_path: Path, project_root: Path) -> List[Path]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dirs = payload.get("run_dirs") if isinstance(payload, dict) else None
    if not isinstance(run_dirs, list):
        raise ValueError("Manifest must include run_dirs list.")
    paths: List[Path] = []
    for entry in run_dirs:
        p = Path(str(entry))
        resolved = p if p.is_absolute() else (project_root / p).resolve()
        paths.append(resolved)
    return paths


def _summarize(values: np.ndarray) -> Dict[str, float]:
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _write_summary_csv(records: List[RunRecord], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_name",
                "run_dir",
                "run_dir_leaf",
                "run_dir_relative",
                "metric_key",
                "metric_value",
                "params_json",
                "timestamp_utc",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r.run_name,
                    r.run_dir,
                    r.run_dir_leaf,
                    r.run_dir_relative,
                    r.metric_key,
                    f"{r.metric_value:.10g}",
                    r.params_json,
                    r.timestamp_utc,
                ]
            )


def _plot_box_with_points(values: np.ndarray, metric_key: str, out_path: Path) -> Dict[str, float]:
    _academic_style()

    stats = _summarize(values)
    n = stats["n"]
    mean = stats["mean"]
    std = stats["std"]
    vmin = stats["min"]
    vmax = stats["max"]

    fig, ax = plt.subplots(figsize=(9.2, 3.4), dpi=300)

    ax.boxplot(
        values,
        vert=False,
        patch_artist=True,
        widths=0.5,
        boxprops={
            "facecolor": "#4C72B0",
            "alpha": 0.18,
            "edgecolor": "#4C72B0",
            "linewidth": 1.6,
        },
        medianprops={"color": "#4C72B0", "linewidth": 2.0},
        whiskerprops={"color": "#4C72B0", "linewidth": 1.4},
        capprops={"color": "#4C72B0", "linewidth": 1.4},
        flierprops={
            "marker": "o",
            "markersize": 4,
            "markerfacecolor": "#4C72B0",
            "alpha": 0.5,
            "markeredgecolor": "none",
        },
    )

    # Jittered scatter points over the boxplot
    # Use RandomState for compatibility with older NumPy versions.
    rng = np.random.RandomState(0)
    jitter = rng.uniform(-0.06, 0.06, size=values.shape[0])
    y = 1.0 + jitter
    ax.scatter(values, y, color="#C44E52", s=38, alpha=0.9, zorder=3, edgecolors="none")

    # Light grey dashed vertical grid lines
    ax.grid(True, axis="x", linestyle="--", linewidth=0.8, alpha=0.35, color="#999999")
    ax.grid(False, axis="y")

    title = f"{metric_key} (N={n})\nMean = {mean:.2f} \u00b1 {std:.2f}"
    ax.set_title(title)
    ax.set_xlabel(_metric_xlabel(metric_key))
    ax.set_yticks([])

    # Annotate min/max values
    xmin, xmax = ax.get_xlim()
    span = xmax - xmin if xmax > xmin else 1.0
    ax.text(vmin - 0.02 * span, 1.0, f"Min: {vmin:.2f}", va="center", ha="right")
    ax.text(vmax + 0.02 * span, 1.0, f"Max: {vmax:.2f}", va="center", ha="left")

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic runs summarizer + academic boxplot tool.")
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="results/runs",
        help="Directory containing run folders.",
    )
    parser.add_argument(
        "--include-glob",
        action="append",
        default=[],
        help="Glob(s) to include run directories (can be passed multiple times).",
    )
    parser.add_argument(
        "--exclude-glob",
        action="append",
        default=[],
        help="Glob(s) to exclude run directories (can be passed multiple times).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="test.r2",
        help="Metric key (dot path) inside metrics.json, e.g., test.r2 or val.rmse.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional manifest JSON listing run_dirs to summarize.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/runs/boxplot.png",
        help="Output path for the boxplot image.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="results/runs/summary.csv",
        help="Output path for the CSV summary.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    runs_dir = (project_root / args.runs_dir).resolve()
    out_path = (project_root / args.out).resolve()
    out_csv = (project_root / args.out_csv).resolve()

    if args.manifest:
        manifest_path = (project_root / args.manifest).resolve()
        run_dirs = _load_manifest_paths(manifest_path, project_root)
        records, skipped = _collect_records_from_paths(
            run_dirs=run_dirs,
            metric_key=args.metric,
            project_root=project_root,
        )
    else:
        records, skipped = _collect_records(
            runs_dir=runs_dir,
            metric_key=args.metric,
            include_globs=list(args.include_glob or []),
            exclude_globs=list(args.exclude_glob or []),
        )

    if not records:
        raise RuntimeError(f"No valid runs found for metric {args.metric!r} under {runs_dir}.")

    values = np.asarray([r.metric_value for r in records], dtype=float)

    _write_summary_csv(records, out_csv)
    stats = _plot_box_with_points(values, args.metric, out_path)

    print(
        "Summary:",
        {
            "n": stats["n"],
            "mean": stats["mean"],
            "std": stats["std"],
            "min": stats["min"],
            "max": stats["max"],
        },
    )
    print("Included runs:")
    for r in records:
        print(f"- {r.run_name}")

    print("Skipped summary:")
    print(skipped)

    print("CSV saved:")
    print(out_csv)
    print("Plot saved:")
    print(out_path)

    print("Example command:")
    print(
        "python src\\summarize_runs.py --runs-dir results/runs --metric test.r2 "
        "--include-glob \"*svr_tuned*\" --out results/runs/stability_test_r2.png"
    )


if __name__ == "__main__":
    main()
