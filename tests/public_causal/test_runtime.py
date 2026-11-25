"""Focused runtime tests; scientific tests import the same exported notebook core."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from PublicCausal.notebook_runtime import (
    DEFAULT_LOCK_GATES, CheckpointStore, RunLogger, WorkerProgressLogger, atomic_write_json,
    check_disk_budget, export_notebook_core, freeze_design_lock,
    inject_parameters, load_notebook_core, stable_hash, validate_design_lock, file_hash,
)


def fixture_notebook() -> dict:
    return {
        "nbformat": 4, "nbformat_minor": 5, "metadata": {},
        "cells": [
            {"id": "parameters", "cell_type": "code", "metadata": {"tags": ["public-causal-parameters"]}, "source": "STUDY_CONFIG = {'phase': 'smoke', 'resume': False, 'custom_default': 17}\n", "outputs": [], "execution_count": None},
            {"id": "core-imports", "cell_type": "code", "metadata": {"tags": ["public-causal-core"]}, "source": "from __future__ import annotations\n", "outputs": [], "execution_count": None},
            {"id": "core-function", "cell_type": "code", "metadata": {"tags": ["public-causal-core"]}, "source": "def twice(value):\n    return value * 2\n", "outputs": [], "execution_count": None},
            {"id": "not-exported", "cell_type": "code", "metadata": {}, "source": "raise RuntimeError('analysis must not execute on import')\n", "outputs": [], "execution_count": None},
        ],
    }


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "src").mkdir()
        self.notebook = self.root / "results" / "models" / "study.ipynb"
        atomic_write_json(self.notebook, fixture_notebook())

    def tearDown(self):
        self.temporary.cleanup()

    def test_tagged_export_is_importable_and_content_addressed(self):
        module_path, digest = export_notebook_core(self.notebook, self.root / "runtime")
        module = load_notebook_core(self.notebook, self.root / "runtime")
        self.assertEqual(module.twice(9), 18)
        self.assertEqual(module.CORE_SOURCE_HASH, digest)
        self.assertIn(digest[:24], module_path.name)
        self.assertEqual(export_notebook_core(self.notebook, self.root / "runtime"), (module_path, digest))
        # Non-core narrative/output changes do not invalidate scientific definitions.
        notebook = fixture_notebook()
        notebook["cells"][-1]["source"] = "print('different analysis')"
        atomic_write_json(self.notebook, notebook)
        self.assertEqual(export_notebook_core(self.notebook, self.root / "runtime")[1], digest)
        notebook["cells"][2]["source"] = "def twice(value):\n    return value * 3\n"
        atomic_write_json(self.notebook, notebook)
        self.assertNotEqual(export_notebook_core(self.notebook, self.root / "runtime")[1], digest)

    def test_modified_generated_source_is_rejected(self):
        path, _ = export_notebook_core(self.notebook, self.root / "runtime")
        path.write_text("print('tampered')")
        with self.assertRaisesRegex(ValueError, "modified runtime"):
            export_notebook_core(self.notebook, self.root / "runtime")

    def test_parameters_are_valid_python_and_exactly_one_cell(self):
        notebook = fixture_notebook()
        configured = inject_parameters(notebook, {"resume": True, "missing": None})
        namespace = {}
        exec(configured["cells"][0]["source"], namespace)
        self.assertEqual(namespace["STUDY_CONFIG"], {"resume": True, "missing": None})
        self.assertEqual(notebook["cells"][0]["execution_count"], None)
        notebook["cells"].append(notebook["cells"][0])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            inject_parameters(notebook, {})

    def test_checkpoint_resume_hashes_seeds_and_rejects_corruption(self):
        store = CheckpointStore(self.root / "checkpoints", {"core": "v1", "data": "public-v1"})
        key = store.key("bootstrap", seed=41, attempt=2)
        self.assertIsNone(store.load(key))
        store.save(key, {"status": "ok", "estimate": 0.1, "seed": 41})
        self.assertEqual(store.load(key)["seed"], 41)
        self.assertNotEqual(key, store.key("bootstrap", seed=42, attempt=2))
        newer = CheckpointStore(store.root, {"core": "v2", "data": "public-v1"})
        self.assertNotEqual(key, newer.key("bootstrap", seed=41, attempt=2))
        with self.assertRaises(ValueError):
            newer.load(key)
        atomic_write_json(store.root / f"{key}.json", {"key": key, "provenance_hash": store.provenance_hash, "result": "corrupt"})
        with self.assertRaises(ValueError):
            store.load(key)

    def lock_payload(self) -> dict:
        artifact = self.root / 'frozen_design.joblib'
        atomic_write_json(artifact, {'fixture': True})
        return {
            "core_source_hash": stable_hash("core"), "data_manifest_hash": stable_hash("data"),
            "config_hash": stable_hash("config"), "frozen_design_hash": file_hash(artifact),
            'frozen_design_path': str(artifact),
            "development_gates": {gate: {"passed": True} for gate in DEFAULT_LOCK_GATES},
        }

    def test_confirmation_requires_passing_immutable_lock(self):
        lock_path = self.root / "design_lock.json"
        with self.assertRaises(PermissionError):
            validate_design_lock(lock_path)
        payload = self.lock_payload()
        payload["development_gates"]["simulation_validation"] = False
        with self.assertRaisesRegex(PermissionError, "simulation_validation"):
            freeze_design_lock(lock_path, payload)
        payload = self.lock_payload()
        lock = freeze_design_lock(lock_path, payload)
        self.assertTrue(lock["confirmation_allowed"])
        self.assertEqual(validate_design_lock(lock_path, payload["core_source_hash"]), lock)
        self.assertEqual(freeze_design_lock(lock_path, payload), lock)
        with self.assertRaisesRegex(PermissionError, "core changed"):
            validate_design_lock(lock_path, stable_hash("changed"))
        with self.assertRaisesRegex(PermissionError, "overwrite"):
            freeze_design_lock(lock_path, dict(payload, config_hash=stable_hash("changed")))
        atomic_write_json(lock_path, dict(lock, development_gates={}))
        with self.assertRaisesRegex(PermissionError, "checksum"):
            validate_design_lock(lock_path)

    def test_frozen_design_blob_integrity_is_checked_before_holdout_access(self):
        lock_path = self.root / 'design_lock.json'
        payload = self.lock_payload()
        freeze_design_lock(lock_path, payload)
        atomic_write_json(payload['frozen_design_path'], {'fixture': 'changed'})
        with self.assertRaisesRegex(PermissionError, 'artifact changed'):
            validate_design_lock(lock_path)

    def test_logger_progress_failure_traceback_and_worker_isolation(self):
        with contextlib.redirect_stdout(io.StringIO()):
            logger = RunLogger(self.root / "run", "fixture", heartbeat_seconds=0)
            logger.progress("fit", 0, 2)
            logger.progress("fit", 1, 2, fold=1, seed=42)
            logger.heartbeat(stage="fit")
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with logger.stage("failed-fit"):
                    raise RuntimeError("intentional")
            worker = RunLogger(self.root / "run", "fixture", worker_id="123", heartbeat_seconds=0)
            worker.event("fit_complete", fold=2)
            worker.close()
            logger.close()
        events = [json.loads(line) for line in (self.root / "run" / "events.jsonl").read_text().splitlines()]
        failed = next(event for event in events if event["event"] == "stage_failed")
        self.assertIn("RuntimeError: intentional", failed["traceback"])
        self.assertEqual(json.loads((self.root / "run" / "status.json").read_text())["status"], "failed")
        self.assertTrue((self.root / "run" / "logs" / "worker_123.jsonl").is_file())
        self.assertTrue((self.root / "run" / "worker_status" / "worker_123.json").is_file())
        self.assertFalse(any(event["event"] == "fit_complete" for event in events))

    def test_disk_guard(self):
        with patch("PublicCausal.notebook_runtime.shutil.disk_usage") as usage:
            usage.return_value.free = 5
            with self.assertRaisesRegex(OSError, "floor"):
                check_disk_budget(self.root, free_floor_bytes=10, budget_bytes=100000)
            usage.return_value.free = 100000
            with self.assertRaisesRegex(OSError, "budget"):
                check_disk_budget(self.root, free_floor_bytes=10, budget_bytes=1)

    def test_worker_fit_progress_has_bounded_logs_and_live_status(self):
        logger = WorkerProgressLogger(self.root / 'run', 'fixture', 'replicate-12',
                                      interval_seconds=0, trace_budget_bytes=500)
        for fold in range(20):
            logger.event('model_fit', stage='m0', fold=fold, force=True)
        logger.close('completed')
        status = json.loads(logger.status_path.read_text())
        self.assertEqual(status['task_id'], 'replicate-12')
        self.assertEqual(status['status'], 'completed')
        self.assertEqual(status['fit_events_seen'], 22)
        self.assertLess(logger.trace_path.stat().st_size, 1000)

    def test_runner_prepare_resume_keeps_defaults_and_rejects_config_change(self):
        spec = importlib.util.spec_from_file_location("public_causal_runner_test", ROOT / "scripts" / "run_public_causal.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        parser = runner.make_parser()
        arguments = ["--root", str(self.root), "--notebook", str(self.notebook), "--run-id", "fixture", "--prepare-only", "--free-space-floor-gb", "0", "--study-budget-gb", "1"]
        notebook, config, directory = runner.prepare_run(parser.parse_args(arguments))
        self.assertEqual(config["custom_default"], 17)
        self.assertEqual(config["bootstrap_target"], 10)
        self.assertEqual(config["workers"], 4)
        self.assertEqual(config["threads"], 1)
        self.assertTrue((directory / "configured.ipynb").exists())
        with self.assertRaises(FileExistsError):
            runner.prepare_run(parser.parse_args(arguments))
        runner.prepare_run(parser.parse_args(arguments + ["--resume"]))
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            runner.prepare_run(parser.parse_args(arguments + ["--resume", "--bootstrap-target", "11"]))
        with self.assertRaises(PermissionError):
            runner.prepare_run(parser.parse_args(arguments + ["--phase", "confirmation"]))

    @unittest.skipUnless(importlib.util.find_spec('nbclient'), 'Jupyter required for execution integration')
    def test_runner_executes_array_source_cells_and_saves_outputs(self):
        spec = importlib.util.spec_from_file_location('public_causal_runner_execution_test', ROOT / 'scripts/run_public_causal.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        notebook = fixture_notebook()
        notebook['cells'][-1]['source'] = ['print("array-source-executed")\n', 'assert twice(3) == 6\n']
        for cell in notebook['cells']:
            if isinstance(cell['source'], str):
                cell['source'] = cell['source'].splitlines(keepends=True)
        atomic_write_json(self.notebook, notebook)
        args = runner.make_parser().parse_args(['--root', str(self.root), '--notebook', str(self.notebook),
            '--run-id', 'execution-fixture', '--free-space-floor-gb', '0', '--study-budget-gb', '1'])
        configured, config, directory = runner.prepare_run(args)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.execute_run(configured, config, directory, args), 0)
        saved = json.loads((directory / 'executed.ipynb').read_text())
        self.assertEqual(saved['cells'][-1]['outputs'][0]['text'], 'array-source-executed\n')
        self.assertEqual(json.loads((directory / 'cell_checkpoint.json').read_text())['cells_completed'], 4)


if __name__ == "__main__":
    unittest.main()
