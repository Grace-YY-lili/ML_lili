# QSAR_Project_v2 — Reproducible QSAR Pipeline

This project is organized to support high-impact journal standards for reproducibility, auditability, and clear provenance of results.

## Dataset
- Source: TODO (e.g., internal dataset / literature / database name and version)
- Raw location: `data/raw/`
- Processed location: `data/processed/`
- Dataset identifier: TODO (e.g., DOI, accession, commit hash, or checksum)
- Inclusion/exclusion criteria: TODO
- Data curation steps: TODO (salt stripping, standardization, deduplication, etc.)

## Target & Units
- Target name: TODO
- Units: TODO
- Transformation(s): TODO (e.g., log10, standardization, clipping)
- Missing-value handling: TODO
- Outlier policy: TODO

## Features (RDKit/CDFT)
- Feature families:
- RDKit descriptors: TODO (list or reference to descriptor set)
- CDFT descriptors: TODO (list or reference to descriptor set)
- Additional features: TODO
- Feature generation code: TODO (e.g., module path under `src/`)
- Feature filtering/selection steps: TODO (variance, correlation, ElasticNet, etc.)
- Final feature set definition: TODO (where it is stored and how it is versioned)

## Splitting & Validation
- Split strategy: TODO (random / scaffold / time-based / grouped)
- Split proportions: TODO (e.g., train/val/test)
- Seeds and repeats: TODO
- Nested CV: TODO (outer/inner folds and rationale)
- Leakage controls: TODO (what is fit only on train, and when)
- Applicability domain (AD): TODO (method and thresholds)
- Y-scrambling / permutation tests: TODO

## Models & Hyperparameters
- Models evaluated: TODO (e.g., SVR, XGBoost, RF, ElasticNet)
- Model implementation details: TODO (library + version)
- Hyperparameter search:
- Search method: TODO (grid / random / Bayesian)
- Search space: TODO
- Selection criterion: TODO (primary metric + tie-breakers)
- Final training protocol: TODO (data used, refit rules)

## Metrics
- Primary metric(s): TODO (e.g., R^2, RMSE, MAE, Q^2)
- Secondary metric(s): TODO
- Confidence intervals / uncertainty: TODO (bootstrap / repeated CV)
- Statistical comparisons: TODO (if multiple models)

## Results Provenance (`results/runs` convention)
All material results should be written to a run-specific directory:

- Run directory pattern: `results/runs/YYYY-MM-DD_<model>_<split>_<seed>/`
- Each run directory should contain at least:
- `config.json` or `config.yaml`: full resolved configuration
- `metrics.json`: all metrics for train/val/test and CV folds
- `features.csv`: final feature list used for that run
- `predictions_<split>.csv`: predictions with IDs/SMILES and ground truth
- `model/`: serialized model artifacts
- `logs/`: relevant logs and diagnostic outputs
- Optional: `figures/` or references to figure scripts

Provenance requirements:
- Every figure/table must point to one or more specific run directories.
- Every run must be reproducible from a single config and fixed seeds.

## How To Reproduce (placeholder commands)
TODO: replace with real commands once the pipeline entrypoints exist.

```powershell
# 1) Create/activate environment
# TODO

# 2) Prepare data
# TODO: python -m src.data.prepare --config configs/data.yaml

# 3) Run feature engineering
# TODO: python -m src.features.build --config configs/features.yaml

# 4) Train and evaluate model
# TODO: python -m src.train --config configs/model_svr.yaml

# 5) Generate figures/tables
# TODO: python -m src.report.figures --run results/runs/TODO
```

## Unified Parity Plotting
The single entry point is `src/plot_from_run.py`. Use `--all` to generate BOTH/CDFT/RDKIT
plots; it will map to the corresponding runs (e.g., featsCDFT/featsRDKIT) when present.
Plots always write to `results/figures` with names:
`<run_leaf>__<TAG>__PARITY.png` where `TAG` ∈ {BOTH, CDFT, RDKIT}. Archive runs are supported
and are auto-located under `results/archive`. Plotting never retrains if predictions exist.

One-shot PowerShell example (latest split888 SVR run → draw all three plots → print paths):
```powershell
$run = Get-ChildItem -Directory -Recurse results\runs,results\archive -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -match 'svr' -and $_.Name -match 'split888' } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 -ExpandProperty FullName
python src\plot_from_run.py --run-dir "$run" --all --out-dir "results/figures"
Write-Host "Figures written under results/figures for run:" $run
```

Verification helper (recompute test R2 from predictions_test.csv and compare to metrics.json):
```powershell
python -c "import json, pathlib, pandas as pd; from sklearn.metrics import r2_score; p=pathlib.Path(r'<BEST_RUN_DIR>'); df=pd.read_csv(p/'predictions_test.csv'); r2=r2_score(df['y_true'], df['y_pred']); m=json.load(open(p/'metrics.json')); print('r2_recomputed', r2); print('r2_metrics', m['test']['r2'])"
```

## Correlation heatmap (Spearman Top-K)
Generate a global Spearman correlation heatmap using merged train/val/test splits
and the Top-K features ranked by |rho| versus the target.

```powershell
python src\plot_spearman_heatmap.py --feature-set both --target-col "log(1o2)" --top-k 30 --out-dir "results/figures"
```

## Environment / Dependencies
- Environment spec: TODO (e.g., `environment.yml` or `requirements.txt`)
- Key dependencies and versions:
- Python: TODO
- RDKit: TODO
- scikit-learn: TODO
- XGBoost / LightGBM / CatBoost: TODO
- pandas / numpy / scipy: TODO
- Reproducibility settings: TODO (threading, seeds, deterministic flags)

## Figure / Table Mapping
Use this section to map manuscript artifacts to exact run directories and scripts.

- Figure 1: TODO (script/module, run directory, inputs)
- Figure 2: TODO
- Figure 3: TODO
- Table 1: TODO
- Table 2: TODO
- Supplementary Figure S1: TODO
- Supplementary Table S1: TODO

---

### Reproducibility Checklist (journal-oriented)
- Data source and version are recorded. TODO
- Split strategy and seeds are recorded. TODO
- Feature generation and selection are fully specified. TODO
- Hyperparameter search space and selection criteria are recorded. TODO
- All results map to run directories with configs and metrics. TODO
