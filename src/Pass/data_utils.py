"""
Shared utilities for data discovery and loading across all model dataloaders.

This module standardizes how data files are discovered and loaded, supporting:
- Recursive discovery from the processed PFF triplet root with season subfolder support
- A shared canonical pass cache that is reusable across Pass model families
- Optional tensor-shard caches for model-specific preprocessing shortcuts
- Parquet-first priority with optional CSV fallback
- Schema validation with clear error messages
"""

import os
import sys
import json
import hashlib
import re
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union, Callable
import pandas as pd
import torch

# Ensure we have access to repo root utilities
PASS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]

# Canonical data root for all models
DEFAULT_DATA_ROOT = "data/processed/pff_match_triplets"
DEFAULT_EVENT_ROOT = "data/raw/event"
DEFAULT_CACHE_ROOT = "data/processed/cache"
CANONICAL_CACHE_VERSION = "v3_velocity_orientation_fix"
PP_PS_CACHE_VERSION = "v5_velocity_orientation_fix"
EPV_CACHE_VERSION = "v3_velocity_orientation_fix"
PFF_ADAPTER_VERSION = "v4_velocity_orientation_fix"
PFF_TRIPLET_REQUIRED_FILES = ("tracking.parquet", "events.parquet", "players.parquet")
DEFAULT_SEASON_SAMPLE_RATIO = 1.0

# Supported pass outcome labels used by model tensorizers
PASS_OUTCOME_ALLOWED = {"C", "D", "B", "O", "S", "G", "I"}
NULL_EQUIVALENT_TOKENS = {"", "nan", "none", "null", "na", "n/a", "nat", "<na>"}


def _clean_null_equivalents(df: pd.DataFrame) -> pd.DataFrame:
    """Replace string null tokens with pandas NA without changing numeric columns."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(
                lambda value: pd.NA
                if value is None or (isinstance(value, str) and value.strip().lower() in NULL_EQUIVALENT_TOKENS)
                else value
            )
    return df

# In-process cache to avoid loading the same game JSON multiple times.
_PASS_OUTCOME_INDEX_CACHE: Dict[Tuple[str, int], pd.DataFrame] = {}

PASS_SOURCE_FORMAT_AUTO = "auto"
PASS_SOURCE_FORMAT_LEGACY = "legacy_wide"
PASS_SOURCE_FORMAT_PFF = "pff_match_triplets"
ACTION_SOURCE_FORMAT_AUTO = PASS_SOURCE_FORMAT_AUTO
ACTION_SOURCE_FORMAT_LEGACY = PASS_SOURCE_FORMAT_LEGACY
ACTION_SOURCE_FORMAT_PFF = PASS_SOURCE_FORMAT_PFF

PFF_PASS_OUTCOME_MAP = {
    "completed": "C",
    "intercepted": "D",
    "blocked": "B",
    "out_of_play": "O",
    "stoppage": "S",
    "shot_at_goal": "I",
    "shot_at_own_goal": "G",
}

# Future carry/ball-drive work should reuse these generic action abstractions.
ActionSource = Dict[str, Any]
ActionCanonicalizer = Callable[[ActionSource], Tuple[pd.DataFrame, Dict[str, Any]]]


def get_data_root(custom_root: Optional[str] = None) -> Path:
    """
    Get the canonical data root, resolved relative to repo root.

    Args:
        custom_root: Optional custom path override. If not absolute, resolved relative to REPO_ROOT.

    Returns:
        Resolved Path to the data directory.
    """
    if custom_root is None:
        root = Path(DEFAULT_DATA_ROOT)
    else:
        root = Path(custom_root)

    if not root.is_absolute():
        root = REPO_ROOT / root

    return root.resolve()


def _resolve_data_path(directory: Union[str, Path]) -> Path:
    resolved = Path(directory)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def _parse_env_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None

    token = value.strip().lower()
    if token in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None


def _parse_env_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None

    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return None


def _cache_enabled(cache_enabled: Optional[bool] = None, default: bool = False) -> bool:
    if cache_enabled is not None:
        return bool(cache_enabled)

    disable_env = _parse_env_bool(os.getenv("PFF_DISABLE_CACHE"))
    if disable_env is True:
        return False

    enabled_env = _parse_env_bool(os.getenv("PFF_CACHE_ENABLED"))
    if enabled_env is not None:
        return enabled_env

    return default


def _extract_numeric_match_id(raw_value: Any) -> Optional[int]:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return None
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, float):
        if pd.isna(raw_value):
            return None
        return int(raw_value)
    text = str(raw_value)
    matches = re.findall(r"(\d+)", text)
    if not matches:
        return None
    return int(matches[-1])


def _extract_season_key(source: ActionSource) -> str:
    tokens: List[str] = []
    source_path = source.get("source_path")
    if source_path is not None:
        tokens.extend(Path(str(source_path)).parts)
    source_name = source.get("source_name")
    if source_name is not None:
        tokens.append(str(source_name))

    patterns = (
        re.compile(r"^\d{2}-\d{2}$"),
        re.compile(r"^\d{4}[_-]\d{4}$"),
        re.compile(r"\b\d{4}[_-]\d{4}\b"),
        re.compile(r"\b\d{2}-\d{2}\b"),
    )

    for token in tokens:
        for pattern in patterns:
            match = pattern.search(token)
            if match:
                return match.group(0)

    return "unknown"


def _resolve_season_sample_ratio(season_sample_ratio: Optional[float]) -> float:
    if season_sample_ratio is not None:
        ratio = float(season_sample_ratio)
    else:
        env_ratio = _parse_env_float(os.getenv("PFF_SEASON_SAMPLE_RATIO"))
        ratio = DEFAULT_SEASON_SAMPLE_RATIO if env_ratio is None else env_ratio

    if not 0.0 < ratio <= 1.0:
        raise ValueError("season_sample_ratio must be greater than 0 and less than or equal to 1.")
    return ratio


def sample_sources_by_season(
    sources: List[ActionSource],
    season_sample_ratio: Optional[float] = None,
    split_seed: int = 42,
) -> Tuple[List[ActionSource], Dict[str, Any]]:
    """Sample the same fraction of logical sources from each season group."""
    ratio = _resolve_season_sample_ratio(season_sample_ratio)
    season_groups: Dict[str, List[int]] = {}
    for index, source in enumerate(sources):
        season_groups.setdefault(_extract_season_key(source), []).append(index)

    if ratio >= 1.0:
        return list(sources), {
            "enabled": False,
            "season_sample_ratio": ratio,
            "source_count_before_sampling": int(len(sources)),
            "source_count_after_sampling": int(len(sources)),
            "seasons": {
                season: {"available": int(len(indices)), "selected": int(len(indices))}
                for season, indices in sorted(season_groups.items())
            },
        }

    selected_indices = set()
    season_summary: Dict[str, Dict[str, int]] = {}
    for season, indices in sorted(season_groups.items()):
        seed_input = f"{int(split_seed)}:{season}"
        season_seed = int(hashlib.sha1(seed_input.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(season_seed)
        shuffled_indices = list(indices)
        rng.shuffle(shuffled_indices)
        selected_count = max(1, int(round(ratio * len(shuffled_indices))))
        selected_count = min(selected_count, len(shuffled_indices))
        selected_indices.update(shuffled_indices[:selected_count])
        season_summary[season] = {
            "available": int(len(indices)),
            "selected": int(selected_count),
        }

    sampled_sources = [source for index, source in enumerate(sources) if index in selected_indices]
    return sampled_sources, {
        "enabled": True,
        "season_sample_ratio": ratio,
        "source_count_before_sampling": int(len(sources)),
        "source_count_after_sampling": int(len(sampled_sources)),
        "seasons": season_summary,
    }


def _normalize_source_format(source_format: Optional[str]) -> str:
    if source_format is None:
        return PASS_SOURCE_FORMAT_AUTO
    token = str(source_format).strip().lower()
    allowed = {
        PASS_SOURCE_FORMAT_AUTO,
        PASS_SOURCE_FORMAT_LEGACY,
        PASS_SOURCE_FORMAT_PFF,
    }
    if token not in allowed:
        raise ValueError(f"Unsupported source_format: {source_format}. Allowed: {sorted(allowed)}")
    return token


def _discover_pff_match_triplet_sources(data_dir: Path) -> List[ActionSource]:
    sources: List[ActionSource] = []
    for candidate in sorted([path for path in data_dir.rglob("*") if path.is_dir()]):
        required_paths = {name: (candidate / name) for name in PFF_TRIPLET_REQUIRED_FILES}
        if not all(path.exists() for path in required_paths.values()):
            continue

        match_id = _extract_numeric_match_id(candidate.name)
        if match_id is None:
            match_id = _extract_numeric_match_id(str(candidate))

        sources.append(
            {
                "source_kind": PASS_SOURCE_FORMAT_PFF,
                "source_format": PASS_SOURCE_FORMAT_PFF,
                "source_name": candidate.name,
                "source_path": str(candidate.resolve()),
                "match_id": match_id,
                "tracking_path": str(required_paths["tracking.parquet"].resolve()),
                "events_path": str(required_paths["events.parquet"].resolve()),
                "players_path": str(required_paths["players.parquet"].resolve()),
            }
        )
    return sources


def discover_pass_sources(
    directory: Union[str, Path],
    source_format: str = PASS_SOURCE_FORMAT_AUTO,
    prefer_parquet: bool = True,
    require_extension: Optional[str] = None,
) -> List[ActionSource]:
    """Discover logical pass sources from legacy files or PFF match-triplet folders."""
    normalized_format = _normalize_source_format(source_format)
    data_dir = _resolve_data_path(directory)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pff_sources = _discover_pff_match_triplet_sources(data_dir)
    pff_component_paths = {
        str(Path(source[path_key]).resolve())
        for source in pff_sources
        for path_key in ("tracking_path", "events_path", "players_path")
        if source.get(path_key) is not None
    }

    legacy_files: List[Path] = []
    if normalized_format in {PASS_SOURCE_FORMAT_AUTO, PASS_SOURCE_FORMAT_LEGACY}:
        try:
            legacy_files = discover_data_files(
                str(data_dir),
                prefer_parquet=prefer_parquet,
                require_extension=require_extension,
            )
            if pff_component_paths:
                legacy_files = [
                    path
                    for path in legacy_files
                    if str(path.resolve()) not in pff_component_paths
                ]
        except FileNotFoundError:
            legacy_files = []

    use_pff = normalized_format == PASS_SOURCE_FORMAT_PFF
    if normalized_format == PASS_SOURCE_FORMAT_AUTO:
        use_pff = len(legacy_files) == 0 and len(pff_sources) > 0

    if use_pff:
        if not pff_sources:
            raise FileNotFoundError(
                f"No PFF match triplet sources found in {data_dir}. "
                f"Expected folders containing {PFF_TRIPLET_REQUIRED_FILES}."
            )
        return pff_sources

    if not legacy_files:
        raise FileNotFoundError(f"No legacy pass data files found in {data_dir}")

    return [
        {
            "source_kind": PASS_SOURCE_FORMAT_LEGACY,
            "source_format": PASS_SOURCE_FORMAT_LEGACY,
            "source_name": path.name,
            "source_path": str(path.resolve()),
            "match_id": _extract_numeric_match_id(path.name),
        }
        for path in legacy_files
    ]


def discover_action_sources(
    directory: Union[str, Path],
    source_format: str = ACTION_SOURCE_FORMAT_AUTO,
    prefer_parquet: bool = True,
    require_extension: Optional[str] = None,
) -> List[ActionSource]:
    """Discover generic action sources using the same adapters/caches as pass discovery."""
    return discover_pass_sources(
        directory=directory,
        source_format=source_format,
        prefer_parquet=prefer_parquet,
        require_extension=require_extension,
    )


def split_sources_by_mode(
    sources: List[ActionSource],
    split_ratio: float,
    split_mode: str = "row",
    split_seed: int = 42,
    split_manifest_path: Optional[Union[str, Path]] = None,
) -> Tuple[List[ActionSource], List[ActionSource], Dict[str, Any]]:
    """Split logical sources for train/validation while preserving backward compatibility.

    row mode: keeps existing behavior by returning all sources as train and no val sources.
    match mode: assigns all samples from each match_id to one split and can persist assignments.
    """
    mode = str(split_mode).strip().lower()
    if mode not in {"row", "match"}:
        raise ValueError("split_mode must be one of: 'row', 'match'")

    if mode == "row":
        return list(sources), [], {
            "split_mode": "row",
            "manifest_path": None,
            "train_source_count": int(len(sources)),
            "val_source_count": 0,
        }

    if not sources:
        return [], [], {
            "split_mode": "match",
            "manifest_path": None,
            "train_source_count": 0,
            "val_source_count": 0,
        }

    manifest_path: Optional[Path] = None
    if split_manifest_path is not None:
        manifest_path = Path(split_manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = (REPO_ROOT / manifest_path).resolve()

    source_keys: List[str] = []
    source_to_key: Dict[str, str] = {}
    for source in sources:
        match_id = source.get("match_id")
        key = str(int(match_id)) if match_id is not None else str(source.get("source_name") or source.get("source_path"))
        source_path = str(source.get("source_path"))
        source_to_key[source_path] = key
        source_keys.append(key)

    unique_keys = sorted(set(source_keys))
    assignment_map: Dict[str, str] = {}

    if manifest_path is not None and manifest_path.exists():
        manifest = load_cache_manifest(manifest_path)
        if manifest is not None:
            stored_assignments = manifest.get("match_assignments", {})
            assignment_map = {str(key): str(value) for key, value in stored_assignments.items()}

    if not assignment_map:
        rng = random.Random(int(split_seed))
        shuffled_keys = list(unique_keys)
        rng.shuffle(shuffled_keys)

        train_count = int(round(float(split_ratio) * len(shuffled_keys)))
        if len(shuffled_keys) > 1:
            train_count = min(max(train_count, 1), len(shuffled_keys) - 1)
        else:
            train_count = len(shuffled_keys)

        train_keys = set(shuffled_keys[:train_count])
        assignment_map = {
            key: ("train" if key in train_keys else "val")
            for key in shuffled_keys
        }

        if manifest_path is not None:
            write_cache_manifest(
                manifest_path,
                {
                    "split_mode": "match",
                    "split_ratio": float(split_ratio),
                    "split_seed": int(split_seed),
                    "match_assignments": assignment_map,
                },
            )

    train_sources: List[ActionSource] = []
    val_sources: List[ActionSource] = []
    for source in sources:
        source_path = str(source.get("source_path"))
        key = source_to_key.get(source_path)
        assignment = assignment_map.get(key, "train")
        if assignment == "val":
            val_sources.append(source)
        else:
            train_sources.append(source)

    return train_sources, val_sources, {
        "split_mode": "match",
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "train_source_count": int(len(train_sources)),
        "val_source_count": int(len(val_sources)),
        "match_count": int(len(unique_keys)),
    }


def split_sources_train_val_test_by_mode(
    sources: List[ActionSource],
    train_ratio: Optional[float] = None,
    split_mode: str = "row",
    split_seed: int = 42,
    split_manifest_path: Optional[Union[str, Path]] = None,
    split_ratio: Optional[float] = None,
    season_sample_ratio: Optional[float] = None,
) -> Tuple[List[ActionSource], List[ActionSource], List[ActionSource], Dict[str, Any]]:
    """Split logical sources into train/validation/test partitions.

    row mode preserves backward compatibility by returning all sources as train.
    match mode assigns all samples from each match_id to train, validation, or test,
    and persists those assignments when a manifest path is provided.
    """
    mode = str(split_mode).strip().lower()
    if mode not in {"row", "match"}:
        raise ValueError("split_mode must be one of: 'row', 'match'")

    if train_ratio is None:
        train_ratio = split_ratio
    if train_ratio is None:
        raise ValueError("split_sources_train_val_test_by_mode requires train_ratio or split_ratio")

    if mode == "row":
        return list(sources), [], [], {
            "split_mode": "row",
            "manifest_path": None,
            "train_source_count": int(len(sources)),
            "val_source_count": 0,
            "test_source_count": 0,
        }

    if not sources:
        return [], [], [], {
            "split_mode": "match_train_val_test",
            "manifest_path": None,
            "train_source_count": 0,
            "val_source_count": 0,
            "test_source_count": 0,
        }

    source_count_before_sampling = int(len(sources))
    sources, season_sampling_summary = sample_sources_by_season(
        sources,
        season_sample_ratio=season_sample_ratio,
        split_seed=split_seed,
    )

    manifest_path: Optional[Path] = None
    if split_manifest_path is not None:
        manifest_path = Path(split_manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = (REPO_ROOT / manifest_path).resolve()

    source_keys: List[str] = []
    source_to_key: Dict[str, str] = {}
    for source in sources:
        match_id = source.get("match_id")
        key = str(int(match_id)) if match_id is not None else str(source.get("source_name") or source.get("source_path"))
        source_path = str(source.get("source_path"))
        source_to_key[source_path] = key
        source_keys.append(key)

    unique_keys = sorted(set(source_keys))
    assignment_map: Dict[str, str] = {}

    if manifest_path is not None and manifest_path.exists():
        manifest = load_cache_manifest(manifest_path)
        if manifest is not None and manifest.get("split_mode") == "match_train_val_test":
            stored_assignments = manifest.get("match_assignments", {})
            candidate_map = {str(key): str(value) for key, value in stored_assignments.items()}
            allowed_values = {"train", "val", "test"}
            if all(candidate_map.get(key) in allowed_values for key in unique_keys):
                assignment_map = {key: candidate_map[key] for key in unique_keys}

    if not assignment_map:
        rng = random.Random(int(split_seed))
        shuffled_keys = list(unique_keys)
        rng.shuffle(shuffled_keys)

        key_count = len(shuffled_keys)
        if key_count == 1:
            train_count = 1
            val_count = 0
            test_count = 0
        elif key_count == 2:
            train_count = 1
            val_count = 0
            test_count = 1
        else:
            train_count = int(round(float(train_ratio) * key_count))
            train_count = min(max(train_count, 1), key_count - 2)
            remaining = key_count - train_count
            val_count = max(1, remaining // 2)
            test_count = remaining - val_count
            if test_count == 0:
                test_count = 1
                val_count = remaining - test_count

        assignment_map = {}
        for index, key in enumerate(shuffled_keys):
            if index < train_count:
                assignment_map[key] = "train"
            elif index < train_count + val_count:
                assignment_map[key] = "val"
            else:
                assignment_map[key] = "test"

        if manifest_path is not None:
            write_cache_manifest(
                manifest_path,
                {
                    "split_mode": "match_train_val_test",
                    "train_ratio": float(train_ratio),
                    "split_seed": int(split_seed),
                    "match_assignments": assignment_map,
                },
            )

    train_sources: List[ActionSource] = []
    val_sources: List[ActionSource] = []
    test_sources: List[ActionSource] = []
    for source in sources:
        source_path = str(source.get("source_path"))
        key = source_to_key.get(source_path)
        assignment = assignment_map.get(key, "train")
        if assignment == "val":
            val_sources.append(source)
        elif assignment == "test":
            test_sources.append(source)
        else:
            train_sources.append(source)

    return train_sources, val_sources, test_sources, {
        "split_mode": "match_train_val_test",
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "train_source_count": int(len(train_sources)),
        "val_source_count": int(len(val_sources)),
        "test_source_count": int(len(test_sources)),
        "match_count": int(len(unique_keys)),
        "source_count_before_sampling": source_count_before_sampling,
        "source_count_after_sampling": int(len(sources)),
        "season_sampling": season_sampling_summary,
    }


def get_cache_root(custom_root: Optional[str] = None) -> Path:
    """Resolve the cache root relative to the repository root."""
    root = Path(custom_root) if custom_root is not None else Path(DEFAULT_CACHE_ROOT)
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root.resolve()


def discover_data_files(
    directory: str,
    prefer_parquet: bool = True,
    require_extension: Optional[str] = None,
) -> List[Path]:
    """
    Discover data files recursively from a directory.

    Supports recursive discovery of files across season subfolders (2022_2023, 2023_2024, etc.).
    By default, prioritizes Parquet files but includes CSV as fallback for backward compatibility.

    Args:
        directory: Data directory path (can be relative to REPO_ROOT or absolute).
        prefer_parquet: If True, return Parquet files; if False, return CSV files.
        require_extension: Force specific extension ('.parquet', '.csv'). Overrides prefer_parquet.

    Returns:
        Sorted list of discovered file paths.

    Raises:
        FileNotFoundError: If directory does not exist.
    """
    data_dir = Path(directory)
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir

    data_dir = data_dir.resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Determine which extension to search for
    if require_extension:
        extensions = (require_extension,)
    elif prefer_parquet:
        extensions = ('.parquet', '.csv')
    else:
        extensions = ('.csv', '.parquet')

    # Discover files recursively
    files = []
    for ext in extensions:
        files.extend([f for f in data_dir.rglob(f"*{ext}") if '.ignore' not in f.parts])

    if not files:
        raise FileNotFoundError(
            f"No data files found in {data_dir} with extensions {extensions}"
        )

    return sorted(files)


def load_parquet_with_fallback(filepath: Path) -> pd.DataFrame:
    """
    Load a Parquet file, with automatic CSV fallback if file is not valid Parquet.

    Args:
        filepath: Path to the file (with .parquet or .csv extension).

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is neither valid Parquet nor valid CSV.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if filepath.suffix == '.parquet' or filepath.suffix == '.pq':
        try:
            return pd.read_parquet(filepath)
        except Exception as e:
            raise ValueError(f"Failed to read Parquet file {filepath}: {e}")

    elif filepath.suffix == '.csv':
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file {filepath}: {e}")

    else:
        raise ValueError(f"Unsupported file extension: {filepath.suffix}")


def validate_schema(
    df: pd.DataFrame,
    required_columns: List[str],
    source_file: Optional[str] = None,
) -> None:
    """
    Validate that a DataFrame has all required columns.

    Args:
        df: DataFrame to validate.
        required_columns: List of column names that must be present.
        source_file: Optional filename for error reporting.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        source_info = f" in {source_file}" if source_file else ""
        raise ValueError(
            f"Missing required columns{source_info}: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns)}"
        )


def get_output_directory(subdir: str, create: bool = True) -> Path:
    """
    Get or create an output directory under results/.

    Args:
        subdir: Subdirectory name (e.g., 'models', 'loss', 'metrics', 'heatmaps').
        create: If True, create the directory if it doesn't exist.

    Returns:
        Resolved Path to the output directory.
    """
    results_root = REPO_ROOT / "results"
    output_dir = results_root / subdir

    if create:
        output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir.resolve()


def _hash_string(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def _normalize_cache_token(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._") or "cache"


def get_source_fingerprint(source_path: Union[str, Path]) -> str:
    """Create a cheap source fingerprint based on path, size, and mtime."""
    resolved = Path(source_path)
    if not resolved.is_absolute():
        resolved = (REPO_ROOT / resolved).resolve()

    stat = resolved.stat()
    fingerprint_input = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"
    return _hash_string(fingerprint_input, length=16)


def get_optional_fingerprint(source_path: Optional[Union[str, Path]]) -> Optional[str]:
    """Return a source fingerprint only when the referenced path exists."""
    if source_path is None:
        return None

    resolved = Path(source_path)
    if not resolved.is_absolute():
        resolved = (REPO_ROOT / resolved).resolve()
    if not resolved.exists():
        return None
    return get_source_fingerprint(resolved)


def build_sample_key(row: pd.Series, fallback_index: Any) -> str:
    """Create a stable per-row cache key reused across model families."""
    parts = [
        _normalize_id(row.get("game_id")),
        _normalize_id(row.get("game_event_id")),
        _normalize_id(row.get("possession_event_id")),
        _normalize_id(row.get("player_id")),
        _normalize_id(fallback_index),
    ]
    return "__".join("na" if part is None else str(part) for part in parts)


def get_cache_artifact_paths(
    source_path: Union[str, Path],
    cache_family: str,
    artifact_stem: str,
    data_suffix: str,
    cache_root: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """Resolve stable data and manifest paths for a cache artifact tied to one source file."""
    resolved_source = Path(source_path)
    if not resolved_source.is_absolute():
        resolved_source = (REPO_ROOT / resolved_source).resolve()

    family_root = get_cache_root(str(cache_root) if cache_root is not None else None) / cache_family
    try:
        relative_parent = resolved_source.parent.relative_to(REPO_ROOT)
    except ValueError:
        relative_parent = Path("external_sources") / _normalize_cache_token(str(resolved_source.parent))
    source_token = _normalize_cache_token(
        f"{resolved_source.stem}_{_hash_string(str(resolved_source.resolve()), length=8)}"
    )
    artifact_token = _normalize_cache_token(artifact_stem)
    base_dir = family_root / relative_parent / source_token
    data_path = base_dir / f"{artifact_token}{data_suffix}"
    manifest_path = base_dir / f"{artifact_token}.json"
    return data_path, manifest_path


def load_cache_manifest(manifest_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    manifest = Path(manifest_path)
    if not manifest.exists():
        return None
    with manifest.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_cache_manifest(manifest_path: Union[str, Path], manifest: Dict[str, Any]) -> Path:
    target = Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return target


def _is_cache_manifest_current_with_fingerprint(
    manifest: Optional[Dict[str, Any]],
    expected_fingerprint: str,
    cache_version: str,
    dependencies: Optional[Dict[str, Any]] = None,
) -> bool:
    if manifest is None:
        return False
    if manifest.get("cache_version") != cache_version:
        return False
    if manifest.get("source_fingerprint") != expected_fingerprint:
        return False
    expected_dependencies = dependencies or {}
    return manifest.get("dependencies", {}) == expected_dependencies


def is_cache_manifest_current(
    manifest: Optional[Dict[str, Any]],
    source_path: Union[str, Path],
    cache_version: str,
    dependencies: Optional[Dict[str, Any]] = None,
) -> bool:
    if manifest is None:
        return False
    if manifest.get("cache_version") != cache_version:
        return False
    if manifest.get("source_fingerprint") != get_source_fingerprint(source_path):
        return False

    expected_dependencies = dependencies or {}
    return manifest.get("dependencies", {}) == expected_dependencies


def load_or_build_canonical_cache(
    source_path: Union[str, Path],
    required_columns: List[str],
    source_filename: Optional[str] = None,
    event_root: Optional[str] = None,
    cache_root: Optional[Union[str, Path]] = None,
    cache_enabled: Optional[bool] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load a canonical enriched dataframe from cache, or build it on miss."""
    source = Path(source_path)
    if not source.is_absolute():
        source = (REPO_ROOT / source).resolve()

    dependency_state = {
        "event_root": str(Path(event_root).as_posix()) if event_root is not None else DEFAULT_EVENT_ROOT,
    }

    if not _cache_enabled(cache_enabled, default=True):
        df = load_parquet_with_fallback(source)
        validate_schema(df, required_columns, source_file=source_filename or source.name)
        enriched_df, merge_summary = enrich_pass_outcome_type(
            df,
            source_filename=source_filename or source.name,
            event_root=event_root,
            source_path=source,
        )
        return enriched_df, {
            "cache_hit": False,
            "cache_data_path": None,
            "cache_manifest_path": None,
            "source_path": str(source),
            "rows_total": int(len(enriched_df)),
            "merge_summary": merge_summary,
        }

    canonical_data_path, manifest_path = get_cache_artifact_paths(
        source_path=source,
        cache_family="canonical",
        artifact_stem="enriched_passes",
        data_suffix=".parquet",
        cache_root=cache_root,
    )
    manifest = load_cache_manifest(manifest_path)

    if canonical_data_path.exists() and is_cache_manifest_current(
        manifest,
        source_path=source,
        cache_version=CANONICAL_CACHE_VERSION,
        dependencies=dependency_state,
    ):
        cached_df = pd.read_parquet(canonical_data_path)
        return cached_df, {
            "cache_hit": True,
            "cache_data_path": str(canonical_data_path),
            "cache_manifest_path": str(manifest_path),
            "source_path": str(source),
            "rows_total": int(len(cached_df)),
            "merge_summary": manifest.get("merge_summary", {}),
        }

    df = load_parquet_with_fallback(source)
    validate_schema(df, required_columns, source_file=source_filename or source.name)
    enriched_df, merge_summary = enrich_pass_outcome_type(
        df,
        source_filename=source_filename or source.name,
        event_root=event_root,
        source_path=source,
    )

    canonical_data_path.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.to_parquet(canonical_data_path, index=False)
    manifest_payload = {
        "cache_version": CANONICAL_CACHE_VERSION,
        "source_path": str(source),
        "source_fingerprint": get_source_fingerprint(source),
        "rows_total": int(len(enriched_df)),
        "dependencies": dependency_state,
        "merge_summary": merge_summary,
        "cache_data_path": str(canonical_data_path),
    }
    write_cache_manifest(manifest_path, manifest_payload)

    return enriched_df, {
        "cache_hit": False,
        "cache_data_path": str(canonical_data_path),
        "cache_manifest_path": str(manifest_path),
        "source_path": str(source),
        "rows_total": int(len(enriched_df)),
        "merge_summary": merge_summary,
    }


def _normalize_processed_outcome_token(value: Any) -> Optional[str]:
    if value is None:
        return None
    token = str(value).strip()
    if not token:
        return None
    lowered = token.lower()
    if lowered in NULL_EQUIVALENT_TOKENS:
        return None

    upper = token.upper()
    if upper in PASS_OUTCOME_ALLOWED:
        return upper

    normalized = lowered.replace(" ", "_")
    return PFF_PASS_OUTCOME_MAP.get(normalized)


def _build_pff_source_fingerprint(source: ActionSource, source_format: str) -> str:
    tracking_fp = get_source_fingerprint(source["tracking_path"])
    events_fp = get_source_fingerprint(source["events_path"])
    players_fp = get_source_fingerprint(source["players_path"])
    data_version = source.get("data_version")
    fingerprint_input = "|".join(
        [
            str(source.get("source_path")),
            tracking_fp,
            events_fp,
            players_fp,
            f"adapter={PFF_ADAPTER_VERSION}",
            f"data_version={data_version}",
            f"source_format={source_format}",
        ]
    )
    return _hash_string(fingerprint_input, length=32)


def _normalize_set_piece_token(value: Any) -> Optional[str]:
    if value is None:
        return None
    token = str(value).strip().lower()
    if not token or token in NULL_EQUIVALENT_TOKENS:
        return None
    mapping = {
        "open_play": "open_play",
        "corner": "corner",
        "free_kick": "free_kick",
        "throw_in": "throw_in",
        "penalty": "penalty",
        "goal_kick": "goal_kick",
        "kick_off": "kick_off",
    }
    if token in mapping:
        return mapping[token]
    compact_mapping = {
        "o": "open_play",
        "c": "corner",
        "f": "free_kick",
        "t": "throw_in",
        "p": "penalty",
        "g": "goal_kick",
        "k": "kick_off",
    }
    return compact_mapping.get(token)


def _build_wide_snapshots_from_tracking(actions_df: pd.DataFrame, tracking_df: pd.DataFrame) -> pd.DataFrame:
    if actions_df.empty:
        return actions_df

    required_tracking_columns = {"match_id", "frame_id", "x", "y", "team_id", "player_id"}
    missing = required_tracking_columns - set(tracking_df.columns)
    if missing:
        raise ValueError(
            f"PFF tracking dataframe missing required columns for wide snapshot pivot: {sorted(missing)}"
        )

    normalized_tracking = tracking_df.copy()
    normalized_tracking["match_id"] = pd.to_numeric(normalized_tracking["match_id"], errors="coerce").astype("Int64")
    normalized_tracking["frame_id"] = pd.to_numeric(normalized_tracking["frame_id"], errors="coerce").astype("Int64")
    normalized_tracking["team_id"] = pd.to_numeric(normalized_tracking["team_id"], errors="coerce").astype("Int64")
    normalized_tracking["player_id"] = pd.to_numeric(normalized_tracking["player_id"], errors="coerce").astype("Int64")

    normalized_tracking = normalized_tracking.dropna(
        subset=["match_id", "frame_id", "team_id", "player_id", "x", "y"]
    )

    grouped_frames: Dict[Tuple[int, int], pd.DataFrame] = {}
    for (match_id, frame_id), frame_group in normalized_tracking.groupby(["match_id", "frame_id"], sort=False):
        grouped_frames[(int(match_id), int(frame_id))] = frame_group.sort_values(
            by=["team_id", "player_id"], kind="mergesort"
        )

    wide_rows: List[Dict[str, Any]] = []
    for row_idx, action in actions_df.iterrows():
        match_id = _normalize_id(action.get("game_id"))
        frame_id = _normalize_id(action.get("frame_id"))
        if match_id is None or frame_id is None:
            continue

        frame_players = grouped_frames.get((match_id, frame_id))
        if frame_players is None or frame_players.empty:
            continue

        wide_row: Dict[str, Any] = {"_row_index": int(row_idx)}
        for player_slot, (_, player) in enumerate(frame_players.iterrows(), start=1):
            suffix = str(player_slot)
            wide_row[f"x_player_{suffix}"] = float(player["x"])
            wide_row[f"y_player_{suffix}"] = float(player["y"])
            wide_row[f"team_id_player_{suffix}"] = int(player["team_id"])
            wide_row[f"original_pId_player_{suffix}"] = int(player["player_id"])

            if "vx" in frame_players.columns:
                vx_value = player.get("vx")
                if pd.notna(vx_value):
                    wide_row[f"vx_player_{suffix}"] = float(vx_value)
            if "vy" in frame_players.columns:
                vy_value = player.get("vy")
                if pd.notna(vy_value):
                    wide_row[f"vy_player_{suffix}"] = float(vy_value)

        wide_rows.append(wide_row)

    if not wide_rows:
        return actions_df

    wide_df = pd.DataFrame(wide_rows)
    merged = actions_df.copy()
    merged["_row_index"] = merged.index.astype(int)
    merged = merged.merge(wide_df, on="_row_index", how="left")
    merged = merged.drop(columns=["_row_index"])
    return merged


def _canonicalize_pff_triplet_source(source: ActionSource) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    events_df = _clean_null_equivalents(pd.read_parquet(source["events_path"]))
    tracking_df = _clean_null_equivalents(pd.read_parquet(source["tracking_path"]))

    if "match_id" not in events_df.columns:
        raise ValueError("PFF events.parquet missing required column: match_id")
    if "event_id" not in events_df.columns:
        raise ValueError("PFF events.parquet missing required column: event_id")
    if "possession_id" not in events_df.columns:
        raise ValueError("PFF events.parquet missing required column: possession_id")
    if "possession_type" not in events_df.columns:
        raise ValueError("PFF events.parquet missing required column: possession_type")

    events = events_df.copy()
    events["possession_type"] = events["possession_type"].astype(str).str.lower().str.strip()
    pass_events = events[events["possession_type"] == "pass"].copy()

    if pass_events.empty:
        empty_df = pd.DataFrame(
            columns=[
                "game_id",
                "game_event_id",
                "possession_event_id",
                "player_id",
                "ball_x_start",
                "ball_y_start",
                "ball_x_end",
                "ball_y_end",
                "team_id",
                "pass_outcome_type",
                "frame_id",
            ]
        )
        return empty_df, {
            "rows_total": 0,
            "rows_pass_events": 0,
            "rows_missing_outcome": 0,
            "rows_kept": 0,
        }

    pass_events["game_id"] = pd.to_numeric(pass_events["match_id"], errors="coerce").astype("Int64")
    pass_events["game_event_id"] = pd.to_numeric(pass_events["event_id"], errors="coerce").astype("Int64")
    pass_events["possession_event_id"] = pd.to_numeric(pass_events["possession_id"], errors="coerce").astype("Int64")
    pass_events["player_id"] = pd.to_numeric(pass_events.get("player_id"), errors="coerce").astype("Int64")
    pass_events["team_id"] = pd.to_numeric(pass_events.get("team_id"), errors="coerce").astype("Int64")

    frame_source_col = "start_frame_id" if "start_frame_id" in pass_events.columns else "frame_id"
    pass_events["frame_id"] = pd.to_numeric(pass_events.get(frame_source_col), errors="coerce").astype("Int64")

    ball_x = pd.to_numeric(pass_events.get("ball_x"), errors="coerce")
    ball_y = pd.to_numeric(pass_events.get("ball_y"), errors="coerce")

    pass_events["ball_x_start"] = pd.to_numeric(pass_events.get("ball_x_start"), errors="coerce")
    pass_events["ball_y_start"] = pd.to_numeric(pass_events.get("ball_y_start"), errors="coerce")
    pass_events["ball_x_end"] = pd.to_numeric(pass_events.get("ball_x_end"), errors="coerce")
    pass_events["ball_y_end"] = pd.to_numeric(pass_events.get("ball_y_end"), errors="coerce")

    # Start-coordinate fallbacks are acceptable because event x/y is the ball carrier
    # location. End-coordinate fallbacks to ball_x or x/y are not acceptable: they
    # leak the start location into the pass-selection target. Endpoints must come from
    # processed ball end tracking or target/receiver player location in dataset.py.
    pass_events["ball_x_start"] = pass_events["ball_x_start"].fillna(ball_x)
    pass_events["ball_y_start"] = pass_events["ball_y_start"].fillna(ball_y)

    if "x" in pass_events.columns:
        pass_events["ball_x_start"] = pass_events["ball_x_start"].fillna(
            pd.to_numeric(pass_events["x"], errors="coerce")
        )
    if "y" in pass_events.columns:
        pass_events["ball_y_start"] = pass_events["ball_y_start"].fillna(
            pd.to_numeric(pass_events["y"], errors="coerce")
        )

    pass_events["pass_outcome_type"] = pass_events.get("pass_outcome", pd.Series(index=pass_events.index))
    if "cross_outcome" in pass_events.columns:
        pass_events["pass_outcome_type"] = pass_events["pass_outcome_type"].where(
            pass_events["pass_outcome_type"].notna(),
            pass_events["cross_outcome"],
        )
    pass_events["pass_outcome_type"] = pass_events["pass_outcome_type"].map(_normalize_processed_outcome_token)

    if "set_piece" in pass_events.columns:
        pass_events["set_piece_normalized"] = pass_events["set_piece"].map(_normalize_set_piece_token)

    rows_total = int(len(events))
    rows_pass = int(len(pass_events))
    rows_missing_outcome = int(pass_events["pass_outcome_type"].isna().sum())

    endpoint_distance = (
        (pass_events["ball_x_end"] - pass_events["ball_x_start"]) ** 2
        + (pass_events["ball_y_end"] - pass_events["ball_y_start"]) ** 2
    ) ** 0.5
    rows_missing_endpoint = int(pass_events[["ball_x_end", "ball_y_end"]].isna().any(axis=1).sum())
    rows_endpoint_near_start = int(((endpoint_distance <= 0.5) & endpoint_distance.notna()).sum())

    canonical_df = pass_events.dropna(
        subset=[
            "game_id",
            "game_event_id",
            "possession_event_id",
            "player_id",
            "team_id",
            "frame_id",
            "ball_x_start",
            "ball_y_start",
            "ball_x_end",
            "ball_y_end",
            "pass_outcome_type",
        ]
    ).copy()

    canonical_endpoint_distance = (
        (canonical_df["ball_x_end"] - canonical_df["ball_x_start"]) ** 2
        + (canonical_df["ball_y_end"] - canonical_df["ball_y_start"]) ** 2
    ) ** 0.5
    canonical_df = canonical_df[canonical_endpoint_distance > 0.5].copy()

    canonical_df = _build_wide_snapshots_from_tracking(canonical_df, tracking_df)

    canonical_df["game_id"] = canonical_df["game_id"].astype("Int64")
    canonical_df["game_event_id"] = canonical_df["game_event_id"].astype("Int64")
    canonical_df["possession_event_id"] = canonical_df["possession_event_id"].astype("Int64")
    canonical_df["player_id"] = canonical_df["player_id"].astype("Int64")
    canonical_df["team_id"] = canonical_df["team_id"].astype("Int64")

    summary = {
        "rows_total": rows_total,
        "rows_pass_events": rows_pass,
        "rows_missing_outcome": rows_missing_outcome,
        "rows_missing_endpoint": rows_missing_endpoint,
        "rows_endpoint_near_start": rows_endpoint_near_start,
        "rows_kept": int(len(canonical_df)),
    }
    return canonical_df, summary


def load_or_build_canonical_pass_cache(
    source: Union[ActionSource, str, Path],
    required_columns: List[str],
    source_filename: Optional[str] = None,
    event_root: Optional[str] = None,
    cache_root: Optional[Union[str, Path]] = None,
    source_format: str = PASS_SOURCE_FORMAT_AUTO,
    cache_enabled: Optional[bool] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load canonical pass rows from legacy file or PFF match-triplet source."""
    normalized_format = _normalize_source_format(source_format)
    cache_is_enabled = _cache_enabled(cache_enabled, default=True)

    if isinstance(source, dict):
        logical_source = dict(source)
    else:
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = (REPO_ROOT / source_path).resolve()
        logical_source = {
            "source_kind": PASS_SOURCE_FORMAT_LEGACY,
            "source_format": PASS_SOURCE_FORMAT_LEGACY,
            "source_name": source_filename or source_path.name,
            "source_path": str(source_path),
            "match_id": _extract_numeric_match_id(source_path.name),
        }

    source_kind = logical_source.get("source_kind", PASS_SOURCE_FORMAT_LEGACY)

    if source_kind == PASS_SOURCE_FORMAT_LEGACY:
        return load_or_build_canonical_cache(
            source_path=logical_source["source_path"],
            required_columns=required_columns,
            source_filename=source_filename or logical_source.get("source_name"),
            event_root=event_root,
            cache_root=cache_root,
            cache_enabled=cache_is_enabled,
        )

    if source_kind != PASS_SOURCE_FORMAT_PFF:
        raise ValueError(f"Unsupported source_kind for canonical loader: {source_kind}")

    if not cache_is_enabled:
        canonical_df, merge_summary = _canonicalize_pff_triplet_source(logical_source)
        validate_schema(
            canonical_df,
            required_columns + ["pass_outcome_type"],
            source_file=source_filename or logical_source.get("source_name"),
        )
        return canonical_df, {
            "cache_hit": False,
            "cache_data_path": None,
            "cache_manifest_path": None,
            "source_path": str(logical_source["source_path"]),
            "source_kind": PASS_SOURCE_FORMAT_PFF,
            "rows_total": int(len(canonical_df)),
            "merge_summary": merge_summary,
        }

    canonical_data_path, manifest_path = get_cache_artifact_paths(
        source_path=logical_source["source_path"],
        cache_family="canonical",
        artifact_stem="enriched_passes",
        data_suffix=".parquet",
        cache_root=cache_root,
    )
    manifest = load_cache_manifest(manifest_path)

    dependency_state = {
        "source_kind": PASS_SOURCE_FORMAT_PFF,
        "adapter_version": PFF_ADAPTER_VERSION,
        "source_format": normalized_format,
        "events_path": logical_source.get("events_path"),
        "tracking_path": logical_source.get("tracking_path"),
        "players_path": logical_source.get("players_path"),
        "data_version": logical_source.get("data_version"),
    }
    expected_fingerprint = _build_pff_source_fingerprint(logical_source, normalized_format)

    if canonical_data_path.exists() and _is_cache_manifest_current_with_fingerprint(
        manifest=manifest,
        expected_fingerprint=expected_fingerprint,
        cache_version=CANONICAL_CACHE_VERSION,
        dependencies=dependency_state,
    ):
        cached_df = pd.read_parquet(canonical_data_path)
        return cached_df, {
            "cache_hit": True,
            "cache_data_path": str(canonical_data_path),
            "cache_manifest_path": str(manifest_path),
            "source_path": str(logical_source["source_path"]),
            "source_kind": PASS_SOURCE_FORMAT_PFF,
            "rows_total": int(len(cached_df)),
            "merge_summary": manifest.get("merge_summary", {}),
        }

    canonical_df, merge_summary = _canonicalize_pff_triplet_source(logical_source)
    validate_schema(
        canonical_df,
        required_columns + ["pass_outcome_type"],
        source_file=source_filename or logical_source.get("source_name"),
    )

    canonical_data_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_df.to_parquet(canonical_data_path, index=False)
    manifest_payload = {
        "cache_version": CANONICAL_CACHE_VERSION,
        "source_path": str(logical_source["source_path"]),
        "source_fingerprint": expected_fingerprint,
        "rows_total": int(len(canonical_df)),
        "dependencies": dependency_state,
        "merge_summary": merge_summary,
        "cache_data_path": str(canonical_data_path),
    }
    write_cache_manifest(manifest_path, manifest_payload)

    return canonical_df, {
        "cache_hit": False,
        "cache_data_path": str(canonical_data_path),
        "cache_manifest_path": str(manifest_path),
        "source_path": str(logical_source["source_path"]),
        "source_kind": PASS_SOURCE_FORMAT_PFF,
        "rows_total": int(len(canonical_df)),
        "merge_summary": merge_summary,
    }


def load_tensor_cache(
    source_path: Union[str, Path],
    cache_family: str,
    artifact_stem: str,
    cache_version: str,
    dependencies: Optional[Dict[str, Any]] = None,
    cache_root: Optional[Union[str, Path]] = None,
    cache_enabled: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Load a tensor shard if its manifest is still valid for the source file."""
    if not _cache_enabled(cache_enabled, default=False):
        return None

    data_path, manifest_path = get_cache_artifact_paths(
        source_path=source_path,
        cache_family=cache_family,
        artifact_stem=artifact_stem,
        data_suffix=".pt",
        cache_root=cache_root,
    )
    manifest = load_cache_manifest(manifest_path)
    if not data_path.exists() or not is_cache_manifest_current(manifest, source_path, cache_version, dependencies):
        return None

    payload = torch.load(data_path, map_location="cpu")
    payload["cache_data_path"] = str(data_path)
    payload["cache_manifest_path"] = str(manifest_path)
    payload["cache_hit"] = True
    return payload


def save_tensor_cache(
    source_path: Union[str, Path],
    cache_family: str,
    artifact_stem: str,
    cache_version: str,
    payload: Dict[str, Any],
    dependencies: Optional[Dict[str, Any]] = None,
    cache_root: Optional[Union[str, Path]] = None,
    cache_enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    """Persist a tensor shard and a matching manifest for cache-first dataloaders."""
    if not _cache_enabled(cache_enabled, default=False):
        result = dict(payload)
        result["cache_data_path"] = None
        result["cache_manifest_path"] = None
        result["cache_hit"] = False
        return result

    data_path, manifest_path = get_cache_artifact_paths(
        source_path=source_path,
        cache_family=cache_family,
        artifact_stem=artifact_stem,
        data_suffix=".pt",
        cache_root=cache_root,
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, data_path)

    manifest_payload = {
        "cache_version": cache_version,
        "source_path": str(Path(source_path)),
        "source_fingerprint": get_source_fingerprint(source_path),
        "dependencies": dependencies or {},
        "cache_data_path": str(data_path),
        "sample_count": int(len(payload.get("labels", []))),
    }
    write_cache_manifest(manifest_path, manifest_payload)

    result = dict(payload)
    result["cache_data_path"] = str(data_path)
    result["cache_manifest_path"] = str(manifest_path)
    result["cache_hit"] = False
    return result


def _save_dataframe_to_source(df: pd.DataFrame, source_path: Union[str, Path]) -> Path:
    """
    Persist an enriched dataframe back to its original file path.

    Supports writing Parquet and CSV files, preserving the source extension.
    Existing files are overwritten.
    """
    output_path = Path(source_path)
    if not output_path.is_absolute():
        output_path = (REPO_ROOT / output_path).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix in {".parquet", ".pq"}:
        df.to_parquet(output_path, index=False)
    elif suffix == ".csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(
            f"Unsupported output extension for enriched save: {output_path.suffix}"
        )

    return output_path


def _normalize_id(value: Any) -> Optional[int]:
    """Normalize mixed numeric identifiers (int/float/str) to int or None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    if float(numeric).is_integer():
        return int(numeric)
    return int(round(numeric))


def _normalize_outcome_token(value: Any) -> Optional[str]:
    """Normalize pass outcome token to the canonical uppercase code or None."""
    if value is None:
        return None
    token = str(value).strip()
    if not token:
        return None
    if token.lower() in NULL_EQUIVALENT_TOKENS:
        return None
    token = token.upper()
    if token not in PASS_OUTCOME_ALLOWED:
        return None
    return token


def _extract_game_id(pass_df: pd.DataFrame, source_filename: Optional[str] = None) -> Optional[int]:
    """Extract game_id from dataframe first, then fallback to the filename pattern."""
    if "game_id" in pass_df.columns:
        values = pd.to_numeric(pass_df["game_id"], errors="coerce").dropna()
        if not values.empty:
            return int(values.iloc[0])

    if source_filename:
        numbers = re.findall(r"(\d+)", source_filename)
        if numbers:
            return int(numbers[-1])

    return None


def find_event_file(game_id: int, event_root: Optional[str] = None) -> Path:
    """
    Locate the raw event JSON file for a game id under the event root.
    """
    root = Path(event_root) if event_root is not None else Path(DEFAULT_EVENT_ROOT)
    if not root.is_absolute():
        root = REPO_ROOT / root
    root = root.resolve()

    candidates = sorted(root.glob(f"**/{game_id}.json"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find raw event JSON for game_id={game_id} under {root}."
        )
    return candidates[0]


def _load_pass_outcome_index(game_id: int, event_root: Optional[str] = None) -> pd.DataFrame:
    """
    Build (and cache) a per-game pass outcome index from raw event JSON.

    Returned columns:
    - game_event_id (Int64)
    - possession_event_id (Int64)
    - team_id (Int64)
    - pass_outcome_type (str)
    """
    root = Path(event_root) if event_root is not None else Path(DEFAULT_EVENT_ROOT)
    if not root.is_absolute():
        root = REPO_ROOT / root
    root = root.resolve()

    cache_key = (str(root), int(game_id))
    if cache_key in _PASS_OUTCOME_INDEX_CACHE:
        return _PASS_OUTCOME_INDEX_CACHE[cache_key]

    event_file = find_event_file(game_id=game_id, event_root=str(root))
    with event_file.open("r", encoding="utf-8") as input_file:
        records = json.load(input_file)

    rows: List[Dict[str, Any]] = []
    for record in records:
        game_events = record.get("GAME_EVENTS") or {}
        possession_events = record.get("POSSESSION_EVENTS") or {}

        pass_outcome_type = _normalize_outcome_token(possession_events.get("PASS_OUTCOME_TYPE"))
        if pass_outcome_type is None:
            continue

        rows.append(
            {
                "game_event_id": _normalize_id(record.get("GAME_EVENT_ID")),
                "possession_event_id": _normalize_id(record.get("POSSESSION_EVENT_ID")),
                "team_id": _normalize_id(game_events.get("TEAM_ID")),
                "pass_outcome_type": pass_outcome_type,
            }
        )

    if not rows:
        index_df = pd.DataFrame(
            columns=["game_event_id", "possession_event_id", "team_id", "pass_outcome_type"]
        )
        _PASS_OUTCOME_INDEX_CACHE[cache_key] = index_df
        return index_df

    index_df = pd.DataFrame(rows)
    for column in ["game_event_id", "possession_event_id", "team_id"]:
        index_df[column] = pd.to_numeric(index_df[column], errors="coerce").astype("Int64")

    _PASS_OUTCOME_INDEX_CACHE[cache_key] = index_df
    return index_df


def enrich_pass_outcome_type(
    pass_df: pd.DataFrame,
    source_filename: Optional[str] = None,
    event_root: Optional[str] = None,
    source_path: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fill/normalize `pass_outcome_type` in pass tracking data using raw event JSON.

    Strategy priority:
    1. Keep already valid `pass_outcome_type` values if present.
    2. Fill by (game_event_id, possession_event_id).
    3. Fill by (game_event_id, team_id).
    4. Fill by (possession_event_id, team_id).
    5. Fill by unique game_event_id mapping.
    6. Fill by unique possession_event_id mapping.

    Returns:
        (enriched_dataframe, merge_summary)

    Args:
        pass_df: Pass tracking dataframe.
        source_filename: Optional filename for game_id extraction.
        event_root: Optional event root override.
        source_path: Optional path to persist the enriched dataframe.
            If provided, the dataframe is saved to this same path.
    """
    enriched_df = pass_df.copy()

    required_merge_columns = ["game_event_id", "possession_event_id", "team_id"]
    missing_merge_columns = [col for col in required_merge_columns if col not in enriched_df.columns]
    if missing_merge_columns:
        raise ValueError(
            "Missing required columns for pass outcome merge: "
            f"{missing_merge_columns}."
        )

    if "pass_outcome_type" not in enriched_df.columns:
        enriched_df["pass_outcome_type"] = pd.NA

    game_id = _extract_game_id(enriched_df, source_filename=source_filename)
    if game_id is None:
        raise ValueError(
            "Could not resolve game_id for pass outcome merge. "
            "Expected column 'game_id' or filename with numeric game id."
        )

    event_index = _load_pass_outcome_index(game_id=game_id, event_root=event_root)

    for column in required_merge_columns:
        enriched_df[column] = pd.to_numeric(enriched_df[column], errors="coerce").astype("Int64")

    enriched_df["pass_outcome_type"] = enriched_df["pass_outcome_type"].map(_normalize_outcome_token)

    rows_total = int(len(enriched_df))
    rows_existing = int(enriched_df["pass_outcome_type"].notna().sum())

    rows_pair = 0
    rows_game_team = 0
    rows_possession_team = 0
    rows_game_unique = 0
    rows_possession_unique = 0

    def _fill_from_lookup(lookup_df: pd.DataFrame, on_columns: List[str]) -> int:
        nonlocal enriched_df
        missing_mask = enriched_df["pass_outcome_type"].isna()
        if not missing_mask.any() or lookup_df.empty:
            return 0

        subset = enriched_df.loc[missing_mask, on_columns].copy().reset_index()
        merged = subset.merge(
            lookup_df[on_columns + ["pass_outcome_type"]],
            on=on_columns,
            how="left",
        )
        merged = merged.dropna(subset=["pass_outcome_type"])
        if merged.empty:
            return 0

        enriched_df.loc[merged["index"], "pass_outcome_type"] = merged["pass_outcome_type"].values
        return int(len(merged))

    if not event_index.empty:
        pair_lookup = (
            event_index.dropna(subset=["game_event_id", "possession_event_id", "pass_outcome_type"])
            .drop_duplicates(subset=["game_event_id", "possession_event_id"], keep="first")
        )
        rows_pair = _fill_from_lookup(pair_lookup, ["game_event_id", "possession_event_id"])

        game_team_lookup = (
            event_index.dropna(subset=["game_event_id", "team_id", "pass_outcome_type"])
            .drop_duplicates(subset=["game_event_id", "team_id"], keep="first")
        )
        rows_game_team = _fill_from_lookup(game_team_lookup, ["game_event_id", "team_id"])

        possession_team_lookup = (
            event_index.dropna(subset=["possession_event_id", "team_id", "pass_outcome_type"])
            .drop_duplicates(subset=["possession_event_id", "team_id"], keep="first")
        )
        rows_possession_team = _fill_from_lookup(possession_team_lookup, ["possession_event_id", "team_id"])

        game_unique_counts = (
            event_index.dropna(subset=["game_event_id", "pass_outcome_type"])
            .groupby("game_event_id")["pass_outcome_type"]
            .nunique()
        )
        game_unique_ids = game_unique_counts[game_unique_counts == 1].index
        game_unique_lookup = (
            event_index[event_index["game_event_id"].isin(game_unique_ids)]
            .dropna(subset=["game_event_id", "pass_outcome_type"])
            .drop_duplicates(subset=["game_event_id"], keep="first")
        )
        rows_game_unique = _fill_from_lookup(game_unique_lookup, ["game_event_id"])

        possession_unique_counts = (
            event_index.dropna(subset=["possession_event_id", "pass_outcome_type"])
            .groupby("possession_event_id")["pass_outcome_type"]
            .nunique()
        )
        possession_unique_ids = possession_unique_counts[possession_unique_counts == 1].index
        possession_unique_lookup = (
            event_index[event_index["possession_event_id"].isin(possession_unique_ids)]
            .dropna(subset=["possession_event_id", "pass_outcome_type"])
            .drop_duplicates(subset=["possession_event_id"], keep="first")
        )
        rows_possession_unique = _fill_from_lookup(possession_unique_lookup, ["possession_event_id"])

    rows_filled = int(enriched_df["pass_outcome_type"].notna().sum()) - rows_existing
    rows_missing_after_merge = int(enriched_df["pass_outcome_type"].isna().sum())

    saved_to: Optional[Path] = None
    if source_path is not None:
        saved_to = _save_dataframe_to_source(enriched_df, source_path)

    summary: Dict[str, Any] = {
        "game_id": int(game_id),
        "rows_total": rows_total,
        "rows_existing": rows_existing,
        "rows_filled": rows_filled,
        "rows_missing_after_merge": rows_missing_after_merge,
        "filled_from_pair": rows_pair,
        "filled_from_game_event_team": rows_game_team,
        "filled_from_possession_event_team": rows_possession_team,
        "filled_from_game_event_unique": rows_game_unique,
        "filled_from_possession_event_unique": rows_possession_unique,
        "saved_to": str(saved_to) if saved_to is not None else None,
    }
    return enriched_df, summary
