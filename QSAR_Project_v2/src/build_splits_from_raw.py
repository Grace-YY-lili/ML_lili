from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def _simple_yaml_parse(text: str) -> Dict[str, Any]:
    """Parse a minimal YAML subset: `key: value` per line."""
    data: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        try:
            if "." in value:
                data[key] = float(value)
            else:
                data[key] = int(value)
        except ValueError:
            data[key] = value
    return data


def load_split_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        cfg = yaml.safe_load(text) or {}
        if not isinstance(cfg, dict):
            raise ValueError("Split config did not parse to a mapping.")
        return cfg
    except Exception:
        return _simple_yaml_parse(text)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower()
    return df


@dataclass(frozen=True)
class SplitConfig:
    strategy: str
    split_seed: int
    train: float
    val: float
    test: float
    smiles_col: str
    target_col: str

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SplitConfig":
        target_col = str(d.get("target_col", "log(1o2)"))
        return SplitConfig(
            strategy=str(d.get("strategy", "random")),
            split_seed=int(d.get("split_seed", 42)),
            train=float(d.get("train", 0.70)),
            val=float(d.get("val", 0.15)),
            test=float(d.get("test", 0.15)),
            smiles_col=str(d.get("smiles_col", "smiles")),
            target_col=target_col.strip().lower(),
        )


def _validate_ratios(train: float, val: float, test: float) -> Tuple[float, float, float]:
    total = train + val + test
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive number.")
    # Normalize to protect against small floating-point mismatches.
    return train / total, val / total, test / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val/test splits from a raw CSV.")
    parser.add_argument(
        "--raw-path",
        type=str,
        default="data/raw/cdft_qsar_cdft+rdkit2d.csv",
        help="Path to raw input CSV.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/split.yaml",
        help="Path to split configuration YAML.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=None,
        help="Explicit target column name; overrides config target_col.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Optional override for split seed without changing the config file.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    raw_path = (project_root / args.raw_path).resolve()
    config_path = (project_root / args.config).resolve()
    processed_dir = project_root / "data" / "processed"

    config_dict = load_split_config(config_path)
    cfg = SplitConfig.from_dict(config_dict)

    if cfg.strategy.lower() != "random":
        raise ValueError(f"Unsupported split strategy: {cfg.strategy!r}. Expected 'random'.")

    effective_split_seed = int(args.split_seed) if args.split_seed is not None else cfg.split_seed

    df = pd.read_csv(raw_path)
    df = _normalize_columns(df)

    target_col = (args.target_col.strip().lower() if args.target_col else cfg.target_col)
    if target_col not in df.columns:
        raise ValueError(
            f"Target column {target_col!r} not found in data columns: {list(df.columns)}"
        )

    train_ratio, val_ratio, test_ratio = _validate_ratios(cfg.train, cfg.val, cfg.test)

    processed_dir.mkdir(parents=True, exist_ok=True)

    dataset_n = int(len(df))
    if dataset_n < 3:
        raise ValueError(f"Dataset too small to split: n={dataset_n}.")

    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=effective_split_seed,
        shuffle=True,
    )

    val_relative = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_relative,
        random_state=effective_split_seed,
        shuffle=True,
    )

    train_path = processed_dir / "train_data.csv"
    val_path = processed_dir / "val_data.csv"
    test_path = processed_dir / "test_data.csv"
    manifest_path = processed_dir / "split_manifest.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    manifest = {
        "data_path": str(raw_path),
        "dataset_n": dataset_n,
        "split_seed": cfg.split_seed,
        "effective_split_seed": effective_split_seed,
        "strategy": cfg.strategy,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "columns": {
            "smiles_col": cfg.smiles_col.strip().lower(),
            "target_col_requested": target_col,
            "target_col_resolved": target_col,
        },
        "sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Wrote splits:")
    print(f"- {train_path}")
    print(f"- {val_path}")
    print(f"- {test_path}")
    print(f"- {manifest_path}")


if __name__ == "__main__":
    main()
