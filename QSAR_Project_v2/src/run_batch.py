from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import subprocess
from pathlib import Path
from typing import Dict, List
import fnmatch
import zipfile

from run_artifacts import load_split_manifest, resolve_run_dir, split_seed_from_manifest
from plot_from_run import plot_from_run_multi

def _parse_bool(text: str) -> bool:
    t = text.strip().lower()
    if t in {"1", "true", "t", "yes", "y"}:
        return True
    if t in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Could not parse boolean value from: {text!r}")


def _parse_split_seeds(spec: str) -> List[int]:
    spec = spec.strip()
    if not spec:
        raise ValueError("--split-seeds cannot be empty.")

    parts = [p.strip() for p in spec.split(",") if p.strip()]
    seeds: List[int] = []

    for part in parts:
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start_s = start_s.strip()
            end_s = end_s.strip()
            if not start_s or not end_s:
                raise ValueError(f"Invalid range token: {part!r}")
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError as exc:
                raise ValueError(f"Invalid seed token: {part!r}") from exc
            if start < 0 or end < 0:
                raise ValueError(f"Seed must be >= 0 in token: {part!r}")
            if end < start:
                raise ValueError(f"Invalid range: {part!r}")
            seeds.extend(list(range(start, end + 1)))
        else:
            try:
                value = int(part)
            except ValueError as exc:
                raise ValueError(f"Invalid seed token: {part!r}") from exc
            if value < 0:
                raise ValueError(f"Seed must be >= 0 in token: {part!r}")
            seeds.append(value)

    # Deduplicate while preserving order
    seen = set()
    ordered: List[int] = []
    for s in seeds:
        if s not in seen:
            ordered.append(s)
            seen.add(s)
    return ordered


def _normalize_models(raw: List[str]) -> List[str]:
    models: List[str] = []
    for item in raw:
        if "," in item:
            models.extend([p.strip() for p in item.split(",") if p.strip()])
        else:
            models.append(item.strip())
    return [m for m in models if m]


def _build_model_commands(
    model_key: str,
    seed: int,
    target_col: str,
    feature_set: str,
    refit_on_trainval: bool,
    robust_select: bool,
    delta: float,
    batch_tag: str,
    use_cv: bool,
    cv_folds: int,
    cv_repeats: int,
    cv_shuffle: bool,
    smooth_search: bool,
    stage2_factor: float,
    stage2_points: int,
    stage2_eps_points: int,
    corr_threshold: float,
    outlier_z: float,
    outlier_on: str,
) -> List[str]:
    if model_key == "svr_base":
        return [
            sys.executable,
            "src/train_svr.py",
            "--seed",
            str(seed),
            "--target-col",
            target_col,
            "--feature-set",
            feature_set,
        ]
    if model_key == "svr_tuned":
        return [
            sys.executable,
            "src/tune_svr_smallgrid.py",
            "--seed",
            str(seed),
            "--target-col",
            target_col,
            "--feature-set",
            feature_set,
            "--run-name",
            batch_tag,
            "--refit-on-trainval",
            str(refit_on_trainval).lower(),
            "--robust-select",
            str(robust_select).lower(),
            "--delta",
            str(delta),
            "--use-cv",
            str(use_cv).lower(),
            "--cv-folds",
            str(cv_folds),
            "--cv-repeats",
            str(cv_repeats),
            "--cv-shuffle",
            str(cv_shuffle).lower(),
            "--smooth-search",
            str(smooth_search).lower(),
            "--stage2-factor",
            str(stage2_factor),
            "--stage2-points",
            str(stage2_points),
            "--stage2-eps-points",
            str(stage2_eps_points),
            "--corr-threshold",
            str(corr_threshold),
            "--outlier-z",
            str(outlier_z),
            "--outlier-on",
            outlier_on,
        ]

    raise ValueError(
        f"Unknown model key: {model_key!r}. Supported keys: svr_base, svr_tuned"
    )


def _write_log(log_path: Path, argv: List[str], result: subprocess.CompletedProcess[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    content = [
        f"argv: {' '.join(argv)}",
        f"returncode: {result.returncode}",
        "--- stdout ---",
        result.stdout or "",
        "--- stderr ---",
        result.stderr or "",
    ]
    log_path.write_text("\n".join(content), encoding="utf-8")


def _run_command(argv: List[str], log_path: Path, dry_run: bool) -> bool:
    if dry_run:
        print("DRY-RUN:", " ".join(argv))
        return True

    result = subprocess.run(argv, capture_output=True, text=True, shell=False)
    _write_log(log_path, argv, result)
    return result.returncode == 0


def _summarize_model(
    model_key: str,
    runs_dir: Path,
    dry_run: bool,
    batch_tag: str,
    manifest_path: Path,
) -> None:
    out_png = runs_dir / f"stability_{model_key}_test_r2_{batch_tag}.png"
    out_csv = runs_dir / f"stability_{model_key}_test_r2_{batch_tag}.csv"

    base_argv = [
        sys.executable,
        "src/summarize_runs.py",
        "--runs-dir",
        str(runs_dir),
        "--metric",
        "test.r2",
        "--out",
        str(out_png),
        "--out-csv",
        str(out_csv),
        "--manifest",
        str(manifest_path),
    ]

    if dry_run:
        print("DRY-RUN:", " ".join(base_argv))
        return

    result = subprocess.run(base_argv, capture_output=True, text=True, shell=False)
    if result.returncode == 0:
        print(result.stdout.strip())
        return

    print("Summarize failed for", model_key)
    print(result.stderr.strip())


def _build_archive_include_globs(models: List[str]) -> List[str]:
    includes: List[str] = []
    for model_key in models:
        if model_key == "svr_tuned":
            includes.append("*_svr_tuned_*")
        elif model_key == "svr_base":
            includes.append("*_svr_*")
        else:
            includes.append(f"*{model_key}*")
    return includes


def _matches_any(name: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def _archive_runs(
    runs_dir: Path,
    models: List[str],
    batch_tag: str,
    include_globs: List[str],
    exclude_globs: List[str],
    dry_run: bool,
) -> None:
    archive_dir = runs_dir.parent / "archive" / f"runs_{batch_tag}"
    zip_path = archive_dir / "batch_runs.zip"

    model_includes = _build_archive_include_globs(models)
    includes = model_includes + list(include_globs or []) + [f"*{batch_tag}*"]
    excludes = list(exclude_globs or []) + ["_batch_logs*", "archive*"]

    run_dirs = [
        p
        for p in runs_dir.iterdir()
        if p.is_dir()
        and "_split" in p.name
        and (not includes or _matches_any(p.name, includes))
        and not _matches_any(p.name, excludes)
    ]

    if not run_dirs:
        print("Archive: no matching run dirs found. Nothing to archive.")
        return

    if dry_run:
        print("DRY-RUN archive target:", archive_dir)
        for p in run_dirs:
            print("DRY-RUN archive include:", p)
        return

    archive_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for run_dir in run_dirs:
            for file_path in run_dir.rglob("*"):
                if file_path.is_file():
                    rel = file_path.relative_to(runs_dir)
                    zf.write(file_path, rel)

    for run_dir in run_dirs:
        dest = archive_dir / run_dir.name
        run_dir.rename(dest)

    print("Archive directory:")
    print(archive_dir)
    print("Archive zip:")
    print(zip_path)


def _snapshot_run_dirs(runs_dir: Path) -> set[str]:
    return {p.name for p in runs_dir.iterdir() if p.is_dir()}


def _diff_new_run_dirs(before: set[str], after: set[str]) -> List[str]:
    return sorted([name for name in after - before if "_split" in name])


def _load_manifest_run_dirs(path: Path, project_root: Path) -> List[Path]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    run_dirs = payload.get("run_dirs") if isinstance(payload, dict) else None
    if not isinstance(run_dirs, list):
        return []
    resolved: List[Path] = []
    for entry in run_dirs:
        p = Path(str(entry))
        resolved.append(p if p.is_absolute() else (project_root / p).resolve())
    return resolved


def _write_manifest(path: Path, batch_tag: str, run_dirs: List[Path], project_root: Path) -> None:
    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(project_root))
        except Exception:
            return str(p)

    payload = {
        "batch_tag": batch_tag,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_dirs": [_rel(p) for p in run_dirs],
        "run_dir_leafs": [p.name for p in run_dirs],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _pick_best_run_from_csv(csv_path: Path, project_root: Path) -> Path:
    import csv as _csv

    best_row = None
    best_value = float("-inf")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            try:
                value = float(row.get("metric_value", "nan"))
            except Exception:
                continue
            if value > best_value:
                best_value = value
                best_row = row
    if not best_row:
        raise RuntimeError(f"No valid rows found in {csv_path}")
    run_dir = best_row.get("run_dir") or best_row.get("run_dir_relative") or ""
    run_leaf = best_row.get("run_dir_leaf") or best_row.get("run_name") or ""
    if not run_dir and not run_leaf:
        raise RuntimeError("Best row missing run_dir/run_dir_relative.")
    if run_dir:
        p = Path(run_dir)
        resolved = p if p.is_absolute() else (project_root / run_dir).resolve()
        if resolved.exists():
            return resolved
    if run_leaf:
        return resolve_run_dir(project_root, run_leaf)
    return resolve_run_dir(project_root, run_dir)


def _zip_run_dirs(run_dirs: List[Path], zip_path: Path, base_dir: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for run_dir in run_dirs:
            if not run_dir.exists():
                continue
            for file_path in run_dir.rglob("*"):
                if file_path.is_file():
                    rel = file_path.relative_to(base_dir)
                    zf.write(file_path, rel)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch runner for split-seed stability experiments.\n"
            "Example:\n"
            "  python src/run_batch.py --split-seeds \"42,88,188,288,388,488,588,688,788,888,988,1088,1188,1288,1388,1488,1588,1688,1788,1888\" "
            "--models svr_tuned --target-col \"log(1o2)\" --feature-set both --seed 42 "
            "--use-cv true --cv-folds 5 --cv-repeats 3 --cv-shuffle true "
            "--smooth-search true --stage2-factor 3.0 --stage2-points 5 --stage2-eps-points 5 "
            "--corr-threshold 0.98 --outlier-z 3.0 --outlier-on y "
            "--refit-on-trainval true --robust-select true --delta 0.02 "
            "--summarize true --archive-after true"
            "--target-col \"log(1o2)\" --feature-set both --seed 42 "
            "--refit-on-trainval true --robust-select true --delta 0.02 --archive-after true"
        )
    )
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
        default="log(1o2)",
        help="Target column name (default log(1o2)).",
    )
    parser.add_argument(
        "--batch-tag",
        type=str,
        default=dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Batch tag for run-name and filtering (default: timestamp).",
    )
    parser.add_argument(
        "--split-seeds",
        type=str,
        required=True,
        help="Split seeds spec, e.g. '1-20', '42,88,188,288', or '1-5,42,88,100-105'.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["svr_tuned"],
        help="Model keys to run (default: svr_tuned).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Model/search seed.")
    parser.add_argument(
        "--feature-set",
        type=str,
        default="both",
        choices=["cdft", "rdkit", "both"],
        help="Feature subset to use: cdft, rdkit, or both.",
    )
    parser.add_argument(
        "--refit-on-trainval",
        type=str,
        default="true",
        help="Refit best params on train+val (true/false).",
    )
    parser.add_argument(
        "--robust-select",
        type=str,
        default="true",
        help="Use robust selection for tuning (true/false).",
    )
    parser.add_argument("--delta", type=float, default=0.02, help="Robust selection delta.")
    parser.add_argument("--use-cv", type=str, default="false", help="Use CV in tuner (true/false).")
    parser.add_argument("--cv-folds", type=int, default=5, help="CV folds.")
    parser.add_argument("--cv-repeats", type=int, default=3, help="CV repeats.")
    parser.add_argument("--cv-shuffle", type=str, default="true", help="Shuffle CV (true/false).")
    parser.add_argument("--smooth-search", type=str, default="false", help="Enable smooth search.")
    parser.add_argument("--stage2-factor", type=float, default=3.0, help="Stage2 range factor.")
    parser.add_argument("--stage2-points", type=int, default=5, help="Stage2 logspace points.")
    parser.add_argument("--stage2-eps-points", type=int, default=5, help="Stage2 eps points.")
    parser.add_argument("--corr-threshold", type=float, default=0.0, help="Corr prune threshold.")
    parser.add_argument("--outlier-z", type=float, default=0.0, help="Outlier z threshold.")
    parser.add_argument("--outlier-on", type=str, choices=["y", "x"], default="y", help="Outlier on y/x.")
    parser.add_argument(
        "--continue-on-error",
        type=str,
        default="true",
        help="Continue on errors (true/false).",
    )
    parser.add_argument(
        "--dry-run",
        type=str,
        default="false",
        help="Print commands only (true/false).",
    )
    parser.add_argument(
        "--summarize",
        type=str,
        default="true",
        help="Summarize after batch run (true/false).",
    )
    parser.add_argument(
        "--archive-after",
        type=str,
        default="false",
        help="Archive matching run directories after batch (true/false).",
    )
    parser.add_argument(
        "--figures-root",
        type=str,
        default="results/figures",
        help="Root directory for parity figures.",
    )
    parser.add_argument(
        "--archive-root",
        type=str,
        default="results/archive",
        help="Archive root under project root (fixed to results/archive).",
    )
    parser.add_argument(
        "--archive-include-glob",
        action="append",
        default=[],
        help="Optional extra include glob(s) for archiving.",
    )
    parser.add_argument(
        "--archive-exclude-glob",
        action="append",
        default=[],
        help="Optional extra exclude glob(s) for archiving.",
    )
    args = parser.parse_args()

    split_seeds = _parse_split_seeds(args.split_seeds)
    print(f"Resolved split seeds: {split_seeds}")
    models = _normalize_models(args.models)

    refit_on_trainval = _parse_bool(args.refit_on_trainval)
    robust_select = _parse_bool(args.robust_select)
    use_cv = _parse_bool(args.use_cv)
    cv_shuffle = _parse_bool(args.cv_shuffle)
    smooth_search = _parse_bool(args.smooth_search)
    continue_on_error = _parse_bool(args.continue_on_error)
    dry_run = _parse_bool(args.dry_run)
    summarize = _parse_bool(args.summarize)
    archive_after = _parse_bool(args.archive_after)

    project_root = Path(__file__).resolve().parents[1]
    runs_dir = project_root / "results" / "runs"
    if str(Path(args.figures_root).as_posix()).strip("/") != "results/figures":
        print("Figures root forced to results/figures (plot_from_run standard).")
    logs_dir = runs_dir / "_batch_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    batch_manifest_all = runs_dir / f"batch_manifest_{args.batch_tag}.json"
    batch_manifests: Dict[str, Path] = {}
    batch_run_dirs: Dict[str, List[Path]] = {m: [] for m in models}
    all_run_dirs: List[Path] = []

    _write_manifest(batch_manifest_all, args.batch_tag, all_run_dirs, project_root)
    for model_key in models:
        manifest_path = runs_dir / f"batch_manifest_{args.batch_tag}_{model_key}.json"
        batch_manifests[model_key] = manifest_path
        _write_manifest(manifest_path, args.batch_tag, batch_run_dirs[model_key], project_root)

    for split_seed in split_seeds:
        print(f"\n=== Split seed {split_seed} ===")

        split_argv = [
            sys.executable,
            "src/build_splits_from_raw.py",
            "--raw-path",
            args.raw_path,
            "--config",
            args.config,
            "--target-col",
            args.target_col,
            "--split-seed",
            str(split_seed),
        ]

        split_log = logs_dir / f"{dt.datetime.now().strftime('%Y-%m-%d')}__split{split_seed}__split.log"
        ok = _run_command(split_argv, split_log, dry_run)
        print(f"Split rebuild: {'OK' if ok else 'FAIL'}")
        if not ok and not continue_on_error:
            print("Stopping due to split error.")
            return
        if ok and not dry_run:
            manifest = load_split_manifest(project_root)
            effective = split_seed_from_manifest(manifest)
            if str(effective) != str(split_seed):
                raise RuntimeError(
                    f"Split manifest seed mismatch: expected {split_seed}, got {effective}"
                )

        for model_key in models:
            print(f"Running model: {model_key}")
            before = _snapshot_run_dirs(runs_dir)
            try:
                model_argv = _build_model_commands(
                    model_key=model_key,
                    seed=args.seed,
                    target_col=args.target_col,
                    feature_set=args.feature_set,
                    refit_on_trainval=refit_on_trainval,
                    robust_select=robust_select,
                    delta=args.delta,
                    batch_tag=args.batch_tag,
                    use_cv=use_cv,
                    cv_folds=int(args.cv_folds),
                    cv_repeats=int(args.cv_repeats),
                    cv_shuffle=cv_shuffle,
                    smooth_search=smooth_search,
                    stage2_factor=float(args.stage2_factor),
                    stage2_points=int(args.stage2_points),
                    stage2_eps_points=int(args.stage2_eps_points),
                    corr_threshold=float(args.corr_threshold),
                    outlier_z=float(args.outlier_z),
                    outlier_on=str(args.outlier_on),
                )
            except Exception as e:
                if continue_on_error:
                    print(f"Model error: {e}")
                    continue
                raise

            log_path = logs_dir / (
                f"{dt.datetime.now().strftime('%Y-%m-%d')}__split{split_seed}__{model_key}.log"
            )
            ok = _run_command(model_argv, log_path, dry_run)
            print(f"{model_key}: {'OK' if ok else 'FAIL'}")
            if not ok and not continue_on_error:
                print("Stopping due to model error.")
                return
            if ok and not dry_run:
                after = _snapshot_run_dirs(runs_dir)
                new_names = _diff_new_run_dirs(before, after)
                new_paths = [runs_dir / name for name in new_names]
                batch_run_dirs[model_key].extend(new_paths)
                all_run_dirs.extend(new_paths)
                _write_manifest(batch_manifest_all, args.batch_tag, all_run_dirs, project_root)
                for key, paths in batch_run_dirs.items():
                    manifest_path = runs_dir / f"batch_manifest_{args.batch_tag}_{key}.json"
                    batch_manifests[key] = manifest_path
                    _write_manifest(manifest_path, args.batch_tag, paths, project_root)

    if summarize or archive_after:
        print("\n=== Summarizing runs ===")
        for model_key in models:
            print(f"Summarizing model: {model_key}")
            manifest_path = batch_manifests.get(model_key)
            if manifest_path is None:
                print("Skipping summarize: missing manifest for", model_key)
                continue
            run_dirs = _load_manifest_run_dirs(manifest_path, project_root)
            if not run_dirs:
                print("Skipping summarize: manifest has no runs for", model_key)
                continue
            _summarize_model(model_key, runs_dir, dry_run, args.batch_tag, manifest_path)
            print(
                f"To plot a best run, use run_dir_leaf/run_dir_relative from summary CSV for batch {args.batch_tag}."
            )

    if summarize and not archive_after and not dry_run:
        print("\n=== Auto-plotting best runs ===")
        for model_key in models:
            csv_path = runs_dir / f"stability_{model_key}_test_r2_{args.batch_tag}.csv"
            if not csv_path.exists():
                continue
            best_run_dir = _pick_best_run_from_csv(csv_path, project_root)
            plot_from_run_multi(
                run_dir_arg=str(best_run_dir),
                out_dir_arg="results/figures",
                tags=["BOTH", "CDFT", "RDKIT"],
                target_col=args.target_col,
                feature_set=args.feature_set,
            )

    if archive_after:
        print("\n=== Archiving runs ===")
        if dry_run:
            print("DRY-RUN: archive-after enabled; skipping archive actions.")
            return
        standard_archive_root = (project_root / "results" / "archive").resolve()
        requested_root = (project_root / args.archive_root).resolve()
        if requested_root != standard_archive_root:
            print(
                f"Archive root forced to {standard_archive_root} "
                f"(received {args.archive_root})."
            )
        archive_root = standard_archive_root
        batch_dir = archive_root / args.batch_tag
        batch_dir.mkdir(parents=True, exist_ok=True)

        summary_files: List[Path] = []
        for model_key in models:
            csv_path = runs_dir / f"stability_{model_key}_test_r2_{args.batch_tag}.csv"
            png_path = runs_dir / f"stability_{model_key}_test_r2_{args.batch_tag}.png"
            if csv_path.exists():
                shutil.copy2(csv_path, batch_dir / csv_path.name)
                summary_files.append(csv_path)
            if png_path.exists():
                shutil.copy2(png_path, batch_dir / png_path.name)

        if summary_files:
            best_csv = summary_files[0]
            best_run_dir = _pick_best_run_from_csv(best_csv, project_root)
            fig_paths = plot_from_run_multi(
                run_dir_arg=str(best_run_dir),
                out_dir_arg="results/figures",
                tags=["BOTH", "CDFT", "RDKIT"],
                target_col=args.target_col,
                feature_set=args.feature_set,
            )
            figures_dest = batch_dir / "figures"
            figures_dest.mkdir(parents=True, exist_ok=True)
            for fig_out in fig_paths:
                shutil.copy2(fig_out, figures_dest / fig_out.name)

        zip_path = batch_dir / "batch_runs.zip"
        _zip_run_dirs(all_run_dirs, zip_path, runs_dir)

        if logs_dir.exists():
            shutil.move(str(logs_dir), str(batch_dir / "_batch_logs"))

        for run_dir in all_run_dirs:
            if run_dir.exists():
                shutil.move(str(run_dir), str(batch_dir / run_dir.name))

        if batch_manifest_all.exists():
            shutil.copy2(batch_manifest_all, batch_dir / batch_manifest_all.name)
        for manifest_path in batch_manifests.values():
            if manifest_path.exists():
                shutil.copy2(manifest_path, batch_dir / manifest_path.name)

        archived_run_dirs = [batch_dir / p.name for p in all_run_dirs]
        archived_manifest = batch_dir / f"batch_manifest_{args.batch_tag}_archived.json"
        _write_manifest(archived_manifest, args.batch_tag, archived_run_dirs, project_root)


if __name__ == "__main__":
    main()
