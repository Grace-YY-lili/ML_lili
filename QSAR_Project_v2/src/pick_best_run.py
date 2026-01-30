from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from run_artifacts import resolve_run_dir


def _parse_params(payload: str) -> Dict[str, object]:
    if not payload:
        return {}
    try:
        return json.loads(payload)
    except Exception:
        return {}


def _pick_best_row(
    csv_path: Path,
    metric_key_filter: Optional[str],
) -> Tuple[Dict[str, str], float]:
    best_row: Dict[str, str] | None = None
    best_value: float = float("-inf")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_key = row.get("metric_key", "")
            if metric_key_filter and metric_key != metric_key_filter:
                continue
            try:
                value = float(row.get("metric_value", "nan"))
            except Exception:
                continue
            if value > best_value:
                best_value = value
                best_row = row

    if best_row is None:
        raise RuntimeError("No valid rows found in stability CSV.")
    return best_row, best_value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pick best run from a stability CSV and print resolved run-dir and plot commands."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to stability CSV (e.g., results/runs/stability_*.csv).",
    )
    parser.add_argument(
        "--metric-key",
        type=str,
        default="test.r2",
        help="Metric key to filter on (default: test.r2).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    csv_path = (project_root / args.csv).resolve()

    row, best_value = _pick_best_row(csv_path, args.metric_key)
    run_dir_raw = row.get("run_dir", "") or row.get("run_dir_relative", "")
    run_dir = resolve_run_dir(project_root, run_dir_raw)

    params_payload = _parse_params(row.get("params_json", ""))
    feature_set = str(params_payload.get("feature_set", "both")).lower()
    target_col = str(params_payload.get("target_col", "log(1o2)"))

    print("Best row metric:", best_value)
    print("Run dir:")
    print(run_dir)
    print("Plot command (all feature sets):")
    print(
        f'python src\\plot_from_run.py --run-dir "{run_dir}" --target-col "{target_col}" --all'
    )


if __name__ == "__main__":
    main()
