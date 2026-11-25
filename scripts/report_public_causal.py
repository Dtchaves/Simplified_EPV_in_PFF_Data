#!/usr/bin/env python3
"""Build and execute a results-report notebook from a run's saved artifacts.

Only the notebook's `public-causal-report` cells run: no refits, no data
acquisition, no confirmation access. The run's executed.ipynb and artifacts are
left untouched; outputs are report.ipynb plus tables/figures in report/.

Example:
  .venv/bin/python scripts/report_public_causal.py \
      --phase-dir data/public_statsbomb_causal/runs/<run-id>/full
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "results/models/public_statsbomb_causal_study.ipynb"
REPORT_TAG = "public-causal-report"


def build_report_notebook(source: dict, phase_dir: Path, out_dir: Path, source_path: Path) -> dict:
    cells = source["cells"]
    title = next(cell for cell in cells if cell["cell_type"] == "markdown")
    setup = next(cell for cell in cells if cell["cell_type"] == "code" and "PROJECT_ROOT =" in "".join(cell["source"]))
    report = [cell for cell in cells if REPORT_TAG in cell.get("metadata", {}).get("tags", [])]
    if not report:
        raise SystemExit(f"{source_path} has no {REPORT_TAG!r} cells")
    closing = [cell for cell in cells if cell["cell_type"] == "markdown"
               and "".join(cell["source"]).startswith("## Interpretation and release boundaries")]
    config = dict(report_phase_dir=str(phase_dir), report_out_dir=str(out_dir), run_dir=str(phase_dir))
    parameters = dict(cell_type="code", id="report-parameters", metadata=dict(tags=["report-parameters"]),
                      execution_count=None, outputs=[], source=[f"STUDY_CONFIG = {config!r}\n"])
    notebook = json.loads(json.dumps(dict(source, cells=[title, setup, parameters, *report, *closing])))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
    notebook["metadata"]["public_causal_report"] = dict(
        phase_dir=str(phase_dir), out_dir=str(out_dir), source_notebook=str(source_path),
        source_notebook_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        role="presentation of saved artifacts; not a scientific execution")
    return notebook


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase-dir", type=Path, required=True, help="Run phase folder, e.g. runs/<id>/full or runs/<id>/pilot")
    parser.add_argument("--output", type=Path, help="Report notebook path (default <phase-dir>/report.ipynb)")
    parser.add_argument("--out-dir", type=Path, help="Tables/figures folder (default <phase-dir>/report)")
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    parser.add_argument("--kernel-name", default="python3")
    parser.add_argument("--cell-timeout", type=int, default=600)
    args = parser.parse_args(argv)
    phase_dir = args.phase_dir.resolve()
    if not phase_dir.is_dir():
        raise SystemExit(f"Phase directory not found: {phase_dir}")
    output = (args.output or phase_dir / "report.ipynb").resolve()
    out_dir = (args.out_dir or phase_dir / "report").resolve()
    if output.name == "executed.ipynb":
        raise SystemExit("Refusing to overwrite the run's executed notebook")
    import nbformat
    from nbclient import NotebookClient
    source_path = args.notebook.resolve()
    notebook = build_report_notebook(json.loads(source_path.read_text()), phase_dir, out_dir, source_path)
    execution = nbformat.reads(json.dumps(notebook), as_version=4)
    os.environ.pop("MPLBACKEND", None)  # the kernel uses the inline backend so figures embed as PNG
    client = NotebookClient(execution, timeout=args.cell_timeout, kernel_name=args.kernel_name, allow_errors=False,
                            resources={"metadata": {"path": str(ROOT)}})
    status = 0
    try:
        client.execute()
    except Exception as exc:  # save the partial notebook for diagnosis
        print(f"Report execution failed: {exc}", file=sys.stderr)
        status = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    nbformat.write(execution, str(temporary))
    temporary.replace(output)
    print(json.dumps(dict(status="completed" if status == 0 else "failed", report=str(output), out_dir=str(out_dir))))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
