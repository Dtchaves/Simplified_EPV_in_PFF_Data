#!/usr/bin/env python3
"""Execute the notebook-owned public study, retaining logs and partial notebooks.

Remote launch (from the project directory, after validation):
  tmux new-session -d -s public-causal \
    '.venv/bin/python -u scripts/run_public_causal.py --phase development --run-id euro-study'
  tmux attach -t public-causal

For an interrupted phase use the same arguments plus --resume. The kernel is
rebuilt from cell zero; scientific content-addressed checkpoints prevent repeated
completed fits. Saving cell outputs alone cannot safely restore Python state.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PublicCausal.notebook_runtime import (
    PARAMETERS_TAG, RunLogger, atomic_write_json, check_disk_budget,
    environment_provenance, export_notebook_core, file_hash, inject_parameters,
    stable_hash, utc_now, validate_design_lock,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("smoke", "pilot", "development", "confirmation", "full", "simulation"), default="smoke")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--profile", choices=("submission", "extended"), default="extended")
    parser.add_argument("--study-root", type=Path, help="Existing dedicated public data folder, independent of staged source")
    parser.add_argument("--reuse-development", type=Path, help="Explicit verified import source; never equivalent to --resume")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--bootstrap-target", "--bootstrap-reps", dest="bootstrap_target", type=int)
    parser.add_argument("--bootstrap-max-attempts", type=int)
    parser.add_argument("--simulation-datasets", type=int)
    parser.add_argument("--simulation-bootstrap-target", type=int)
    parser.add_argument("--null-reps", type=int)
    parser.add_argument("--design-lock", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Export core and configured notebook, without launching a kernel or acquiring data")
    parser.add_argument("--kernel-name", default="python3")
    parser.add_argument("--cell-timeout", type=int, default=-1, help="Seconds per cell; -1 permits long resumable stages")
    parser.add_argument("--free-space-floor-gb", type=float, default=10.0)
    parser.add_argument("--study-budget-gb", type=float, default=8.0)
    return parser


def notebook_defaults(notebook: dict) -> dict:
    cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code" and PARAMETERS_TAG in cell.get("metadata", {}).get("tags", [])]
    if len(cells) != 1:
        raise ValueError("Notebook must contain exactly one public-causal-parameters cell")
    source = cells[0]["source"]
    source = "".join(source) if isinstance(source, list) else source
    for statement in ast.parse(source).body:
        if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "STUDY_CONFIG" for target in statement.targets):
            defaults = ast.literal_eval(statement.value)
            if not isinstance(defaults, dict):
                raise ValueError("STUDY_CONFIG defaults must be a literal dictionary")
            return defaults
    raise ValueError("Parameters cell must assign a literal STUDY_CONFIG dictionary")


def prepare_run(args: argparse.Namespace) -> tuple[dict, dict, Path]:
    root = args.root.resolve()
    if not (root / "src").is_dir():
        raise ValueError(f"Invalid project root: {root}")
    if args.workers < 1 or args.threads < 1:
        raise ValueError("workers and threads must be positive")
    if args.threads != 1:
        raise ValueError("This study requires one numerical thread per worker")
    run_id = args.run_id or ("public-" + utc_now().replace(":", "").replace("+", "_"))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,120}", run_id):
        raise ValueError("run-id must be a safe alphanumeric filename component")
    study_root = (args.study_root or root / "data" / "public_statsbomb_causal").resolve()
    if args.study_root is not None and not study_root.is_dir():
        raise ValueError("Explicit study-root must be an existing public study directory")
    reuse_development = args.reuse_development.resolve() if args.reuse_development else None
    if reuse_development is not None and (not reuse_development.is_dir() or not reuse_development.is_relative_to(study_root)):
        raise ValueError("Development import must be an existing directory under the public study folder")
    check_disk_budget(study_root, int(args.free_space_floor_gb * 1024**3), int(args.study_budget_gb * 1024**3))
    notebook_path = (args.notebook or root / "results" / "models" / "public_statsbomb_causal_study.ipynb").resolve()
    notebook = json.loads(notebook_path.read_text())
    config = notebook_defaults(notebook)
    run_dir = study_root / "runs" / run_id / args.phase
    module_path, core_hash = export_notebook_core(notebook_path, run_dir / "runtime")
    lock_path = (args.design_lock or study_root / "runs" / run_id / "design_lock.json").resolve()
    if not lock_path.is_relative_to(study_root.resolve()):
        raise ValueError("Design lock must be under the dedicated public study folder")
    lock = validate_design_lock(lock_path, expected_core_hash=core_hash, expected_profile=args.profile) if args.phase == "confirmation" else None
    if args.profile == "submission":
        defaults = {"smoke": (3, 1, 0, 1), "pilot": (5, 2, 0, 2)}.get(args.phase, (200, 50, 0, 50))
    else:
        defaults = (10, 2, 10, 3) if args.phase == "smoke" else (1000, 200, 1000, 200)
    target = args.bootstrap_target if args.bootstrap_target is not None else defaults[0]
    simulation_datasets = args.simulation_datasets if args.simulation_datasets is not None else defaults[1]
    simulation_bootstrap = args.simulation_bootstrap_target if args.simulation_bootstrap_target is not None else defaults[2]
    null_reps = args.null_reps if args.null_reps is not None else defaults[3]
    attempts = args.bootstrap_max_attempts if args.bootstrap_max_attempts is not None else max(target, math.ceil(target * 1.5))
    if min(target, simulation_datasets, simulation_bootstrap, null_reps, attempts) < 0 or attempts < target:
        raise ValueError("Replication counts must be nonnegative and maximum attempts >= target")
    config.update({
        "root": str(root), "study_root": str(study_root), "run_id": run_id,
        "profile": args.profile, "candidate_device": args.device,
        "reuse_development": str(reuse_development) if reuse_development is not None else None,
        "implementation_tests_path": str(study_root / "runs" / run_id / "manifests" / "implementation_tests.json"),
        "run_dir": str(run_dir), "phase": args.phase, "workers": args.workers,
        "threads": 1, "bootstrap_target": target, "bootstrap_max_attempts": attempts,
        "simulation_datasets": simulation_datasets,
        "simulation_bootstrap_target": simulation_bootstrap, "null_reps": null_reps,
        "resume": args.resume, "design_lock_path": str(lock_path),
        "core_source_hash": core_hash, "core_module_path": str(module_path),
        "notebook_path": str(notebook_path),
        "free_space_floor_bytes": int(args.free_space_floor_gb * 1024**3),
        "study_budget_bytes": int(args.study_budget_gb * 1024**3),
        "helper_source_hashes": {path.name: file_hash(path) for path in sorted((root / "src" / "PublicCausal").glob("*.py"))},
    })
    environment = environment_provenance()
    config['environment_hash'] = stable_hash({key: environment[key] for key in ('python', 'executable', 'packages')})
    if lock is not None:
        config["design_lock_hash"] = file_hash(lock_path)
    run_hash = stable_hash({key: value for key, value in config.items() if key != "resume"})
    prior_path = run_dir / "run_config.json"
    if prior_path.exists():
        prior = json.loads(prior_path.read_text())
        if prior.get("run_config_hash") != run_hash:
            raise ValueError("Existing run configuration changed; use a new run-id")
        if not args.resume:
            raise FileExistsError("Run already exists; pass --resume or choose a new run-id")
    atomic_write_json(prior_path, {"run_config_hash": run_hash, "config": config})
    execution_provenance = {
        **environment, "run_config_hash": run_hash, "core_source_hash": core_hash,
        "notebook_sha256": file_hash(notebook_path), "config": config,
    }
    original_provenance = run_dir / 'execution_provenance.json'
    if not original_provenance.exists():
        atomic_write_json(original_provenance, execution_provenance)
    atomic_write_json(run_dir / 'last_execution_attempt.json', execution_provenance)
    configured = inject_parameters(notebook, config)
    atomic_write_json(run_dir / "configured.ipynb", configured)
    return configured, config, run_dir


def execute_run(notebook: dict, config: dict, run_dir: Path, args: argparse.Namespace) -> int:
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:
        raise RuntimeError("Headless execution requires nbclient and nbformat in the selected environment") from exc
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PYTHONUNBUFFERED"] = "1"
    # The installed python3 kernelspec uses a relative 'python' executable.
    # Select this runner's interpreter without changing any global kernelspec.
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    os.environ["JOBLIB_TEMP_FOLDER"] = str(run_dir / "temporary")
    os.environ["PYTHONPATH"] = os.pathsep.join((str(Path(config["root"]) / "src"), str(Path(config["core_module_path"]).parent), os.environ.get("PYTHONPATH", "")))
    logger = RunLogger(run_dir / "execution", config["run_id"])
    # from_dict preserves on-disk source arrays; reads normalizes them to the
    # strings nbclient executes, including after parameter injection.
    execution_notebook = nbformat.reads(json.dumps(notebook), as_version=4)
    destination = run_dir / "executed.ipynb"
    counts = {"cells_completed": 0}

    class StreamingNotebookClient(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            if msg.get('msg_type') == 'stream':
                sys.stdout.write(msg.get('content', {}).get('text', ''))
                sys.stdout.flush()
            return super().process_message(msg, cell, cell_index)

    def save_notebook() -> None:
        atomic_write_json(destination, dict(execution_notebook))

    def on_start(cell, cell_index, **kwargs):
        check_disk_budget(config["study_root"], config["free_space_floor_bytes"], config["study_budget_bytes"])
        logger.event("cell_start", stage=f"cell_{cell_index}", cell_index=cell_index, cell_id=cell.get("id"))

    def on_executed(cell, cell_index, **kwargs):
        counts["cells_completed"] += 1
        save_notebook()
        atomic_write_json(run_dir / "cell_checkpoint.json", {
            "cell_index": cell_index, "cell_id": cell.get("id"),
            "cells_completed": counts["cells_completed"], "core_source_hash": config["core_source_hash"],
            "notebook_sha256": file_hash(destination), "utc": utc_now(),
            "resume_policy": "rerun kernel state cells and reuse compatible scientific artifacts",
        })
        logger.event("cell_complete", stage=f"cell_{cell_index}", cell_index=cell_index, **counts)

    client = StreamingNotebookClient(
        execution_notebook, timeout=args.cell_timeout, kernel_name=args.kernel_name,
        allow_errors=False, resources={"metadata": {"path": config["root"]}},
        on_cell_start=on_start, on_cell_executed=on_executed,
    )
    try:
        logger.event("execution_start", phase=config["phase"], workers=config["workers"], threads=1, core_source_hash=config["core_source_hash"])
        client.execute()
        save_notebook()
        phase_result = run_dir / "phase_result.json"
        scientific_status = json.loads(phase_result.read_text()).get("status") if phase_result.exists() else None
        if scientific_status in {"failed", "scientific_failure", "blocked", "unsupported"}:
            logger.close(scientific_status, notebook=str(destination), **counts)
            return 2
        logger.close("completed" if scientific_status == "completed" else "execution_complete", scientific_status=scientific_status, notebook=str(destination), **counts)
        return 0
    except BaseException as exc:
        save_notebook()
        logger.close("failed", exception_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(), notebook=str(destination), **counts)
        raise


def main(argv=None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        notebook, config, run_dir = prepare_run(args)
        if args.prepare_only:
            print(json.dumps({"status": "prepared_only", "run_dir": str(run_dir), "core_source_hash": config["core_source_hash"]}, indent=2), flush=True)
            return 0
        return execute_run(notebook, config, run_dir, args)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
