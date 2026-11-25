"""Small execution/checkpoint infrastructure; no scientific estimators live here."""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Iterator


CORE_TAG = "public-causal-core"
PARAMETERS_TAG = "public-causal-parameters"
DEFAULT_LOCK_GATES = (
    "timing_audit", "leakage_tests", "development_diagnostics",
    "null_checks", "simulation_validation",
)
SUBMISSION_LOCK_GATES = (
    "timing_audit", "leakage_tests", "development_diagnostics",
    "weighted_validation", "design_sanity",
)
SPECIFICATION_FIELDS = (
    "horizon", "model", "backend", "support", "subgroup", "inference",
    "simulation", "budget",
)


def _lock_gates(payload: dict, required_gates=None):
    """Recognized profiles cannot weaken their scientific access contract."""
    profile = payload.get("profile")
    if profile is None:
        return DEFAULT_LOCK_GATES if required_gates is None else tuple(required_gates)
    if profile not in ("submission", "extended"):
        raise PermissionError(f"Unknown design-lock profile: {profile!r}")
    gates = SUBMISSION_LOCK_GATES if profile == "submission" else DEFAULT_LOCK_GATES
    if required_gates is not None and tuple(required_gates) != gates:
        raise PermissionError("Recognized profile gates cannot be overridden")
    if profile == "submission":
        manifest = payload.get("specification_manifest")
        if not isinstance(manifest, dict) or any(not manifest.get(key) for key in SPECIFICATION_FIELDS):
            raise PermissionError("Submission lock requires the complete specification manifest")
        if payload.get("specification_manifest_hash") != stable_hash(manifest):
            raise PermissionError("Submission specification manifest hash does not match")
    return gates


DEFAULT_DISK_FLOOR = 10 * 1024**3
DEFAULT_STUDY_BUDGET = 8 * 1024**3


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted((json_safe(item) for item in value), key=repr)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    return value


def stable_hash(value: Any) -> str:
    payload = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_bytes(path, (json.dumps(json_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def export_notebook_core(notebook_path: str | Path, runtime_dir: str | Path) -> tuple[Path, str]:
    """Export tagged declarations verbatim; workers/tests use this single source.

    Never executes notebook output, parameters, download, or analysis cells.
    Tagged cells must be ordinary Python, with no notebook magics.
    """
    notebook_path = Path(notebook_path).resolve()
    notebook = json.loads(notebook_path.read_text())
    cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code" and CORE_TAG in cell.get("metadata", {}).get("tags", [])]
    if not cells:
        raise ValueError(f"Notebook has no {CORE_TAG!r} definition cells")
    definitions = "\n\n".join(_source(cell).rstrip() for cell in cells) + "\n"
    # Future imports must occur before other statements, including our header.
    tree = ast.parse(definitions, filename=str(notebook_path))
    future_lines = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            future_lines.extend(range(node.lineno, node.end_lineno + 1))
    lines = definitions.splitlines(keepends=True)
    future = "".join(line for index, line in enumerate(lines, 1) if index in future_lines)
    body = "".join(line for index, line in enumerate(lines, 1) if index not in future_lines)
    root = next((parent for parent in notebook_path.parents if (parent / "src").is_dir()), None)
    if root is None:
        raise ValueError("Cannot discover project src directory from notebook location")
    source_hash = stable_hash({"export_schema": 1, "definitions": definitions})
    header = (
        "# GENERATED from notebook public-causal-core cells; do not edit.\n"
        + future
        + "import sys as _runtime_sys\n"
        + f"_runtime_src = {str(root / 'src')!r}\n"
        + "if _runtime_src not in _runtime_sys.path:\n    _runtime_sys.path.insert(0, _runtime_src)\n"
        + f"CORE_SOURCE_HASH = {source_hash!r}\n"
    )
    exported = header + body
    compile(exported, str(notebook_path), "exec")
    destination = Path(runtime_dir) / f"public_causal_core_{source_hash[:24]}.py"
    if destination.exists() and destination.read_text() != exported:
        raise ValueError("Generated module hash collision or modified runtime source")
    if not destination.exists():
        atomic_write_bytes(destination, exported.encode())
    return destination, source_hash


def load_notebook_core(notebook_path: str | Path, runtime_dir: str | Path):
    module_path, _ = export_notebook_core(notebook_path, runtime_dir)
    module_name = module_path.stem
    if str(module_path.parent.resolve()) not in sys.path:
        sys.path.insert(0, str(module_path.parent.resolve()))
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load generated core {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def inject_parameters(notebook: dict, config: dict) -> dict:
    """Replace exactly one parameters cell, preserving all other notebook code."""
    import copy
    result = copy.deepcopy(notebook)
    tagged = [cell for cell in result["cells"] if cell.get("cell_type") == "code" and PARAMETERS_TAG in cell.get("metadata", {}).get("tags", [])]
    if len(tagged) != 1:
        raise ValueError(f"Expected exactly one {PARAMETERS_TAG!r} cell, found {len(tagged)}")
    # repr emits valid Python booleans/None; JSON booleans are not Python literals.
    tagged[0]["source"] = "# Injected by run_public_causal.py; persisted in executed notebook.\nSTUDY_CONFIG = " + repr(json_safe(config)) + "\n"
    tagged[0]["outputs"] = []
    tagged[0]["execution_count"] = None
    return result


def directory_bytes(path: str | Path) -> int:
    total = 0
    if not Path(path).exists():
        return 0
    for directory, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
        for name in files:
            file = Path(directory) / name
            if not file.is_symlink():
                try:
                    total += file.stat().st_size
                except FileNotFoundError:
                    pass  # A concurrent atomic replacement is harmless.
    return total


def check_disk_budget(study_root: str | Path, free_floor_bytes: int = DEFAULT_DISK_FLOOR, budget_bytes: int = DEFAULT_STUDY_BUDGET) -> dict:
    root = Path(study_root).resolve()
    existing = root
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    used = directory_bytes(root)
    if free < free_floor_bytes:
        raise OSError(f"Disk floor breached: {free / 1024**3:.2f} GiB free; require {free_floor_bytes / 1024**3:.2f}")
    if used > budget_bytes:
        raise OSError(f"Study budget breached: {used / 1024**3:.2f} GiB; budget {budget_bytes / 1024**3:.2f}")
    return {"free_bytes": free, "study_bytes": used, "free_floor_bytes": free_floor_bytes, "budget_bytes": budget_bytes}


def _passed(value: Any) -> bool:
    return value is True or (isinstance(value, dict) and value.get("passed") is True)


def validate_design_lock(path: str | Path, expected_core_hash: str | None = None, required_gates=None, expected_profile: str | None = None) -> dict:
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise PermissionError("Confirmation requires an immutable design lock and checksum sidecar")
    expected = sidecar.read_text().strip()
    if len(expected) != 64 or file_hash(path) != expected:
        raise PermissionError("Design-lock checksum does not match; confirmation is blocked")
    lock = json.loads(path.read_text())
    if lock.get("status") != "frozen" or lock.get("phase") != "development" or lock.get("confirmation_allowed") is not True:
        raise PermissionError("Design lock is not a passing frozen development design")
    for field in ("core_source_hash", "data_manifest_hash", "config_hash"):
        if not isinstance(lock.get(field), str) or len(lock[field]) != 64:
            raise PermissionError(f"Design lock missing valid {field}")
    required_gates = _lock_gates(lock, required_gates)
    if expected_profile is not None:
        if expected_profile not in ("submission", "extended") or lock.get("profile", "extended") != expected_profile:
            raise PermissionError("Design-lock profile does not match the requested experiment")
    gates = lock.get("development_gates", {})
    failed = [gate for gate in required_gates if not _passed(gates.get(gate))]
    if failed:
        raise PermissionError(f"Confirmation blocked by development gates: {failed}")
    if expected_core_hash is not None and lock["core_source_hash"] != expected_core_hash:
        raise PermissionError("Scientific core changed after design freeze")
    if 'frozen_design_path' in lock or 'frozen_design_hash' in lock:
        artifact = Path(lock.get('frozen_design_path', ''))
        if not artifact.is_file() or file_hash(artifact) != lock.get('frozen_design_hash'):
            raise PermissionError('Frozen support/model artifact changed after design freeze')
    return lock


def freeze_design_lock(path: str | Path, payload: dict, required_gates=None) -> dict:
    """Freeze passing measured gates; refuse replacement of a prior lock."""
    required_gates = _lock_gates(payload, required_gates)
    path = Path(path)
    if path.exists():
        existing = validate_design_lock(path, required_gates=required_gates)
        for key, value in payload.items():
            if key not in {"frozen_at", "status", "phase", "confirmation_allowed"} and json_safe(existing.get(key)) != json_safe(value):
                raise PermissionError(f"Cannot overwrite immutable design lock: changed {key}")
        return existing
    failed = [gate for gate in required_gates if not _passed(payload.get("development_gates", {}).get(gate))]
    if failed:
        raise PermissionError(f"Cannot freeze failing/unrun development gates: {failed}")
    for field in ("core_source_hash", "data_manifest_hash", "config_hash"):
        if not isinstance(payload.get(field), str) or len(payload[field]) != 64:
            raise ValueError(f"Freeze requires a 64-character {field}")
    lock = dict(payload, status="frozen", phase="development", confirmation_allowed=True, frozen_at=utc_now())
    path.parent.mkdir(parents=True, exist_ok=True)
    claim = path.with_suffix(path.suffix + ".freeze-in-progress")
    fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.close(fd)
        if path.exists():
            raise PermissionError("Another process froze this design")
        atomic_write_json(path, lock)
        atomic_write_bytes(path.with_suffix(path.suffix + ".sha256"), (file_hash(path) + "\n").encode())
    finally:
        claim.unlink(missing_ok=True)
    return validate_design_lock(path, required_gates=required_gates)


class CheckpointStore:
    """Content-addressed JSON checkpoints, including failure/attempt records."""

    def __init__(self, root: str | Path, provenance: dict):
        self.root = Path(root)
        self.provenance_hash = stable_hash(provenance)

    def key(self, stage: str, **inputs: Any) -> str:
        return stable_hash({"schema": 1, "provenance": self.provenance_hash, "stage": stage, "inputs": inputs})

    def load(self, key: str) -> Any | None:
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        envelope = json.loads(path.read_text())
        if envelope.get("provenance_hash") != self.provenance_hash or envelope.get("key") != key or stable_hash(envelope.get("result")) != envelope.get("result_hash"):
            raise ValueError(f"Corrupt or incompatible checkpoint: {path}")
        return envelope["result"]

    def save(self, key: str, result: Any) -> Path:
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError("Checkpoint key must be a SHA256 digest")
        return atomic_write_json(self.root / f"{key}.json", {
            "schema": 1, "key": key, "provenance_hash": self.provenance_hash,
            "result_hash": stable_hash(result), "result": result, "saved_at": utc_now(),
        })


class RunLogger:
    """JSONL + stdout, atomic status and heartbeats; forked workers log separately."""

    def __init__(self, run_dir: str | Path, run_id: str, worker_id: str | None = None, heartbeat_seconds: float = 15):
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.worker_id = worker_id
        self.origin_pid = os.getpid()
        self.started = time.monotonic()
        self.stage_started: dict[str, float] = {}
        self.state: dict = {"status": "running", "stage": "initializing"}
        self._mutex = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if heartbeat_seconds > 0 and worker_id is None:
            self._thread = threading.Thread(target=self._heartbeat_loop, args=(heartbeat_seconds,), daemon=True)
            self._thread.start()

    def _paths(self) -> tuple[Path, Path, Path]:
        worker = self.worker_id or (str(os.getpid()) if os.getpid() != self.origin_pid else None)
        if worker:
            return (self.run_dir / "logs" / f"worker_{worker}.jsonl", self.run_dir / "logs" / f"worker_{worker}.log", self.run_dir / "worker_status" / f"worker_{worker}.json")
        return self.run_dir / "events.jsonl", self.run_dir / "stdout.log", self.run_dir / "status.json"

    def event(self, name: str, **fields: Any) -> dict:
        with self._mutex:
            record = json_safe({"utc": utc_now(), "run_id": self.run_id, "pid": os.getpid(), "worker_id": self.worker_id, "event": name, "elapsed_seconds": time.monotonic() - self.started, **fields})
            structured, human, status = self._paths()
            structured.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, sort_keys=True, allow_nan=False)
            with structured.open("a") as handle:
                handle.write(line + "\n")
            message = f"{record['utc']} [{self.run_id}] {name} " + json.dumps(json_safe(fields), sort_keys=True, allow_nan=False)
            with human.open("a") as handle:
                handle.write(message + "\n")
            print(message, flush=True)
            if "status" in fields:
                self.state["status"] = fields["status"]
            for key in ("stage", "completed", "total", "eta_seconds", "reason"):
                if key in fields:
                    self.state[key] = fields[key]
            atomic_write_json(status, {**self.state, "utc": record["utc"], "run_id": self.run_id, "pid": os.getpid(), "elapsed_seconds": record["elapsed_seconds"], "last_event": name})
            return record

    def progress(self, stage: str, completed: int, total: int | None = None, cached_completed: int = 0, **fields: Any) -> dict:
        if cached_completed < 0 or completed < cached_completed:
            raise ValueError("Progress must include nonnegative cached completions")
        now = time.monotonic()
        fresh_completed = completed - cached_completed
        if fresh_completed == 0 or stage not in self.stage_started:
            self.stage_started[stage] = now
        elapsed = now - self.stage_started[stage]
        eta = elapsed * max(0, total - completed) / fresh_completed if total is not None and fresh_completed > 0 else None
        return self.event("progress", stage=stage, completed=completed, total=total,
                          cached_completed=cached_completed, fresh_completed=fresh_completed,
                          stage_elapsed_seconds=elapsed, eta_seconds=eta, **fields)

    def heartbeat(self, **fields: Any) -> dict:
        stage = fields.pop("stage", self.state.get("stage"))
        return self.event("heartbeat", stage=stage, **fields)

    def _heartbeat_loop(self, seconds: float) -> None:
        while not self._stop.wait(seconds):
            try:
                self.heartbeat()
            except Exception:
                # Do not lose the worker's scientific exception to a logger thread.
                traceback.print_exc()

    @contextlib.contextmanager
    def stage(self, name: str, total: int | None = None) -> Iterator["RunLogger"]:
        self.stage_started[name] = time.monotonic()
        self.event("stage_start", stage=name, total=total, completed=0, status="running")
        try:
            yield self
        except BaseException as exc:
            self.event("stage_failed", stage=name, status="failed", exception_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            raise
        else:
            self.event("stage_complete", stage=name, stage_elapsed_seconds=time.monotonic() - self.stage_started[name])

    def close(self, status: str | None = None, **fields: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if status is not None:
            self.event("run_end", status=status, **fields)


class WorkerProgressLogger:
    """Live atomic fit status and bounded sampled traces, not millions of lines.

    Full replicate seeds/results/failures live in master checkpoints. Workers
    expose the current inner/outer stage even while a dataset takes hours.
    """

    def __init__(self, run_dir, run_id, task_id, interval_seconds=3, trace_budget_bytes=8*1024**2):
        self.run_dir = Path(run_dir)
        self.run_id = str(run_id)
        self.task_id = str(task_id)
        self.pid = os.getpid()
        self.started = time.monotonic()
        self.last_write = self.last_trace = 0.
        self.interval = interval_seconds
        self.trace_budget = trace_budget_bytes
        self.events_seen = 0
        self.status_path = self.run_dir / 'worker_status' / f'worker_{self.pid}.json'
        self.trace_path = self.run_dir / 'logs' / f'worker_{self.pid}.jsonl'
        self.event('worker_start', status='running', force=True)

    def event(self, name, **fields):
        self.events_seen += 1
        now = time.monotonic()
        forced = fields.pop('force', False) or name in ('worker_end', 'worker_failed')
        record = json_safe(dict(utc=utc_now(), run_id=self.run_id, task_id=self.task_id,
                               pid=self.pid, event=name, elapsed_seconds=now-self.started,
                               fit_events_seen=self.events_seen, **fields))
        if forced or now-self.last_write >= self.interval:
            atomic_write_json(self.status_path, record)
            self.last_write = now
        if forced or now-self.last_trace >= 60:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.trace_path.exists() or self.trace_path.stat().st_size < self.trace_budget:
                with self.trace_path.open('a') as handle:
                    handle.write(json.dumps(record, sort_keys=True, allow_nan=False)+'\n')
            self.last_trace = now
        return record

    def close(self, status='completed', **fields):
        self.event('worker_end', status=status, force=True, **fields)


def environment_provenance() -> dict:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "nbclient", "nbformat", "ipykernel", "matplotlib", "pyarrow", "shapely", "dowhy", "econml", "xgboost"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "executable": sys.executable, "packages": packages, "utc": utc_now()}
