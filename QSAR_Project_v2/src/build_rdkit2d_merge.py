from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


# Interpretable RDKit 2D descriptors (no fingerprints)
DESCRIPTOR_FUNCS: Dict[str, callable] = {
    "rdkit_MolWt": Descriptors.MolWt,
    "rdkit_HeavyAtomMolWt": Descriptors.HeavyAtomMolWt,
    "rdkit_MolLogP": Descriptors.MolLogP,
    "rdkit_TPSA": Descriptors.TPSA,
    "rdkit_NumHDonors": Descriptors.NumHDonors,
    "rdkit_NumHAcceptors": Descriptors.NumHAcceptors,
    "rdkit_NumRotatableBonds": Descriptors.NumRotatableBonds,
    "rdkit_RingCount": Descriptors.RingCount,
    "rdkit_FractionCSP3": Descriptors.FractionCSP3,
    "rdkit_MolMR": Descriptors.MolMR,
    "rdkit_HeavyAtomCount": Descriptors.HeavyAtomCount,
    "rdkit_NHOHCount": Descriptors.NHOHCount,
    "rdkit_NOCount": Descriptors.NOCount,
    # A few additional interpretable 2D descriptors
    "rdkit_NumAromaticRings": Descriptors.NumAromaticRings,
    "rdkit_NumAliphaticRings": Descriptors.NumAliphaticRings,
    "rdkit_BalabanJ": Descriptors.BalabanJ,
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower()
    return df


def _resolve_smiles_col(df: pd.DataFrame) -> str:
    candidates = ["smiles", "smile"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"SMILES column not found. Available columns: {list(df.columns)}")


def _compute_rdkit_row(smiles: str) -> Dict[str, float]:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return {name: np.nan for name in DESCRIPTOR_FUNCS}

    values: Dict[str, float] = {}
    for name, func in DESCRIPTOR_FUNCS.items():
        try:
            values[name] = float(func(mol))
        except Exception:
            values[name] = np.nan
    return values


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"

    input_path = raw_dir / "cdft_qsar.csv"
    output_path = raw_dir / "cdft_qsar_cdft+rdkit2d.csv"
    manifest_path = raw_dir / "rdkit2d_manifest.json"

    df = pd.read_csv(input_path)
    df = _normalize_columns(df)

    smiles_col = _resolve_smiles_col(df)

    descriptor_names: List[str] = list(DESCRIPTOR_FUNCS.keys())

    rdkit_rows: List[Dict[str, float]] = []
    rdkit_valid: List[int] = []

    for smi in df[smiles_col].astype(str).tolist():
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        is_valid = int(mol is not None)
        rdkit_valid.append(is_valid)

        if mol is None:
            rdkit_rows.append({name: np.nan for name in descriptor_names})
            continue

        values: Dict[str, float] = {}
        for name, func in DESCRIPTOR_FUNCS.items():
            try:
                values[name] = float(func(mol))
            except Exception:
                values[name] = np.nan
        rdkit_rows.append(values)

    rdkit_df = pd.DataFrame(rdkit_rows)
    rdkit_df["rdkit_valid"] = rdkit_valid

    # Merge RDKit descriptors alongside existing CDFT columns
    merged = pd.concat([df, rdkit_df], axis=1)

    # Do not modify target values; we only add columns and normalize names.
    merged.to_csv(output_path, index=False)

    manifest = {
        "n_rows": int(len(merged)),
        "n_rdkit_valid": int(sum(rdkit_valid)),
        "descriptor_names": descriptor_names,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(output_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
