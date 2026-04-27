"""
Shared utilities for data discovery and loading across all model dataloaders.

This module standardizes how data files are discovered and loaded, supporting:
- Recursive discovery from data/passes root with season subfolder support
- Parquet-first priority with optional CSV fallback
- Schema validation with clear error messages
"""

import os
import sys
import json
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union
import pandas as pd
import torch

# Ensure we have access to repo root utilities
PASS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]

# Canonical data root for all models
DEFAULT_DATA_ROOT = "data/passes"
DEFAULT_EVENT_ROOT = "data/raw/event"
DEFAULT_CACHE_ROOT = "data/processed/cache"
CANONICAL_CACHE_VERSION = "v1"
PP_PS_CACHE_VERSION = "v1"
EPV_CACHE_VERSION = "v1"

# Supported pass outcome labels used by model tensorizers
PASS_OUTCOME_ALLOWED = {"C", "D", "B", "O", "S", "G", "I"}
NULL_EQUIVALENT_TOKENS = {"", "nan", "none", "null", "na", "n/a", "nat"}

# In-process cache to avoid loading the same game JSON multiple times.
_PASS_OUTCOME_INDEX_CACHE: Dict[Tuple[str, int], pd.DataFrame] = {}


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
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load a canonical enriched dataframe from cache, or build it on miss."""
    source = Path(source_path)
    if not source.is_absolute():
        source = (REPO_ROOT / source).resolve()

    canonical_data_path, manifest_path = get_cache_artifact_paths(
        source_path=source,
        cache_family="canonical",
        artifact_stem="enriched_passes",
        data_suffix=".parquet",
        cache_root=cache_root,
    )
    manifest = load_cache_manifest(manifest_path)
    dependency_state = {
        "event_root": str(Path(event_root).as_posix()) if event_root is not None else DEFAULT_EVENT_ROOT,
    }

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


def load_tensor_cache(
    source_path: Union[str, Path],
    cache_family: str,
    artifact_stem: str,
    cache_version: str,
    dependencies: Optional[Dict[str, Any]] = None,
    cache_root: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Load a tensor shard if its manifest is still valid for the source file."""
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
) -> Dict[str, Any]:
    """Persist a tensor shard and a matching manifest for cache-first dataloaders."""
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
