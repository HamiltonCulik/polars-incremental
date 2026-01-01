from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .checkpoints.types import atomic_write_json, list_batch_ids


@dataclass(frozen=True)
class CleanupResult:
    removed_offsets: int
    removed_commits: int
    kept_offsets: int
    kept_commits: int


@dataclass(frozen=True)
class TruncateResult:
    removed_offsets: int
    removed_commits: int
    kept_offsets: int
    kept_commits: int
    keep_through_batch_id: int


@dataclass(frozen=True)
class CheckpointInfo:
    checkpoint_dir: Path
    offsets: int
    commits: int
    latest_offset: int | None
    latest_commit: int | None
    pending: int
    start_offset: dict | None
    schema: list[dict] | None
    snapshot_cache_snapshots: int
    snapshot_cache_deltas: int


def cleanup_checkpoint(
    checkpoint_dir: str | Path,
    *,
    keep_last_n: int | None = None,
    older_than_seconds: float | None = None,
    dry_run: bool = False,
) -> CleanupResult:
    """Remove old offset/commit files.

    - keep_last_n keeps the newest N batch ids (based on union of offsets+commits).
    - older_than_seconds removes files with mtime older than now - threshold.
    """
    checkpoint_path = Path(checkpoint_dir)
    offset_dir = checkpoint_path / "offsets"
    commit_dir = checkpoint_path / "commits"

    offset_ids = list_batch_ids(offset_dir)
    commit_ids = list_batch_ids(commit_dir)
    all_ids = sorted(set(offset_ids + commit_ids))

    keep_ids: set[int] = set()
    if keep_last_n is not None and keep_last_n > 0:
        keep_ids.update(all_ids[-keep_last_n:])

    now = time.time()
    threshold = None
    if older_than_seconds is not None and older_than_seconds >= 0:
        threshold = now - older_than_seconds

    removed_offsets = _cleanup_dir(
        offset_dir, keep_ids=keep_ids, threshold=threshold, dry_run=dry_run
    )
    removed_commits = _cleanup_dir(
        commit_dir, keep_ids=keep_ids, threshold=threshold, dry_run=dry_run
    )

    kept_offsets = len(list_batch_ids(offset_dir)) if not dry_run else len(offset_ids) - removed_offsets
    kept_commits = len(list_batch_ids(commit_dir)) if not dry_run else len(commit_ids) - removed_commits

    return CleanupResult(
        removed_offsets=removed_offsets,
        removed_commits=removed_commits,
        kept_offsets=kept_offsets,
        kept_commits=kept_commits,
    )


def truncate_checkpoint(
    checkpoint_dir: str | Path,
    *,
    keep_through_batch_id: int,
    dry_run: bool = False,
) -> TruncateResult:
    """Remove offset/commit files with batch_id greater than keep_through_batch_id."""
    checkpoint_path = Path(checkpoint_dir)
    offset_dir = checkpoint_path / "offsets"
    commit_dir = checkpoint_path / "commits"

    removed_offsets, kept_offsets = _truncate_dir(
        offset_dir, keep_through_batch_id=keep_through_batch_id, dry_run=dry_run
    )
    removed_commits, kept_commits = _truncate_dir(
        commit_dir, keep_through_batch_id=keep_through_batch_id, dry_run=dry_run
    )

    return TruncateResult(
        removed_offsets=removed_offsets,
        removed_commits=removed_commits,
        kept_offsets=kept_offsets,
        kept_commits=kept_commits,
        keep_through_batch_id=keep_through_batch_id,
    )


def reset_checkpoint_start_offset(checkpoint_dir: str | Path) -> dict | None:
    """Remove persisted start_offset metadata, returning the previous value if present."""
    metadata_path = Path(checkpoint_dir) / "metadata.json"
    payload = _load_metadata(metadata_path)
    if payload is None:
        return None
    previous = payload.pop("start_offset", None)
    if previous is not None:
        _save_metadata(metadata_path, payload)
    return previous


def reset_checkpoint_schema(checkpoint_dir: str | Path) -> list[dict] | None:
    """Remove stored schema metadata, returning the previous schema if present."""
    metadata_path = Path(checkpoint_dir) / "metadata.json"
    payload = _load_metadata(metadata_path)
    if payload is None:
        return None
    previous = payload.pop("schema", None)
    if previous is not None:
        _save_metadata(metadata_path, payload)
    return previous if isinstance(previous, list) else None


def inspect_checkpoint(checkpoint_dir: str | Path) -> CheckpointInfo:
    """Return a summary of checkpoint metadata and batch state."""
    checkpoint_path = Path(checkpoint_dir)
    offset_dir = checkpoint_path / "offsets"
    commit_dir = checkpoint_path / "commits"
    offset_ids = list_batch_ids(offset_dir)
    commit_ids = list_batch_ids(commit_dir)
    latest_offset = offset_ids[-1] if offset_ids else None
    latest_commit = commit_ids[-1] if commit_ids else None
    pending_ids = [batch_id for batch_id in offset_ids if batch_id not in set(commit_ids)]
    metadata_path = checkpoint_path / "metadata.json"
    payload = _load_metadata(metadata_path) or {}
    start_offset = payload.get("start_offset") if isinstance(payload.get("start_offset"), dict) else None
    schema = payload.get("schema") if isinstance(payload.get("schema"), list) else None
    snapshot_dir = checkpoint_path / "snapshot_cache" / "snapshots"
    delta_dir = checkpoint_path / "snapshot_cache" / "deltas"
    snapshot_count = len(list(snapshot_dir.glob("*.json"))) if snapshot_dir.exists() else 0
    delta_count = len(list(delta_dir.glob("*.json"))) if delta_dir.exists() else 0
    return CheckpointInfo(
        checkpoint_dir=checkpoint_path,
        offsets=len(offset_ids),
        commits=len(commit_ids),
        latest_offset=latest_offset,
        latest_commit=latest_commit,
        pending=len(pending_ids),
        start_offset=start_offset,
        schema=schema,
        snapshot_cache_snapshots=snapshot_count,
        snapshot_cache_deltas=delta_count,
    )


def cleanup_snapshot_cache(
    checkpoint_dir: str | Path,
    *,
    keep_snapshots: int | None = 1,
    keep_deltas_since_snapshot: int | None = 0,
    dry_run: bool = False,
) -> CleanupResult:
    """Cleanup snapshot_cache snapshots/deltas for Delta checkpoints."""
    checkpoint_path = Path(checkpoint_dir)
    snapshot_dir = checkpoint_path / "snapshot_cache" / "snapshots"
    delta_dir = checkpoint_path / "snapshot_cache" / "deltas"

    snapshot_versions = _list_cache_versions(snapshot_dir)
    delta_versions = _list_cache_versions(delta_dir)

    keep_snapshot_versions: set[int] = set()
    if keep_snapshots is not None and keep_snapshots > 0:
        keep_snapshot_versions.update(snapshot_versions[-keep_snapshots:])

    removed_snapshots = _cleanup_cache_dir(
        snapshot_dir, keep_versions=keep_snapshot_versions, dry_run=dry_run
    )

    keep_delta_versions: set[int] = set()
    if keep_deltas_since_snapshot is not None and keep_deltas_since_snapshot >= 0:
        if keep_snapshot_versions:
            anchor = max(keep_snapshot_versions)
        else:
            anchor = max(snapshot_versions) if snapshot_versions else None
        if anchor is not None:
            threshold = anchor - keep_deltas_since_snapshot
            keep_delta_versions = {v for v in delta_versions if v >= threshold}
        else:
            keep_delta_versions = set(delta_versions)

    removed_deltas = _cleanup_cache_dir(
        delta_dir, keep_versions=keep_delta_versions, dry_run=dry_run
    )

    kept_snapshots = len(_list_cache_versions(snapshot_dir)) if not dry_run else len(snapshot_versions) - removed_snapshots
    kept_deltas = len(_list_cache_versions(delta_dir)) if not dry_run else len(delta_versions) - removed_deltas

    return CleanupResult(
        removed_offsets=removed_snapshots,
        removed_commits=removed_deltas,
        kept_offsets=kept_snapshots,
        kept_commits=kept_deltas,
    )


def _list_cache_versions(directory: Path) -> list[int]:
    if not directory.exists():
        return []
    versions: list[int] = []
    for path in directory.glob("*.json"):
        try:
            versions.append(int(path.stem))
        except ValueError:
            continue
    return sorted(versions)


def _cleanup_cache_dir(
    directory: Path,
    *,
    keep_versions: set[int],
    dry_run: bool,
) -> int:
    removed = 0
    if not directory.exists():
        return removed
    for path in directory.glob("*.json"):
        try:
            version = int(path.stem)
        except ValueError:
            continue
        if version in keep_versions:
            continue
        if not dry_run:
            path.unlink(missing_ok=True)
        removed += 1
    return removed


def vacuum_delta_table(
    table_path: str | Path,
    *,
    retention_hours: float = 168.0,
    dry_run: bool = False,
    enforce_retention: bool | None = None,
) -> Any:
    """Vacuum a Delta table using deltalake (delta-rs)."""
    if isinstance(retention_hours, float) and retention_hours.is_integer():
        retention_hours = int(retention_hours)
    table = _get_delta_table(table_path)
    kwargs: dict[str, Any] = {"retention_hours": retention_hours, "dry_run": dry_run}
    if enforce_retention is not None:
        kwargs["enforce_retention_duration"] = enforce_retention
    try:
        return table.vacuum(**kwargs)
    except TypeError:
        # Older delta-rs versions may not support enforce_retention_duration.
        kwargs.pop("enforce_retention_duration", None)
        return table.vacuum(**kwargs)


def optimize_delta_table(
    table_path: str | Path,
    *,
    mode: str = "compact",
    z_order_columns: Iterable[str] | None = None,
    partition_filters: list[tuple[str, str, Any]] | list[list[tuple[str, str, Any]]] | None = None,
    target_size: int | None = None,
    max_concurrent_tasks: int | None = None,
    max_spill_size: int | None = None,
    min_commit_interval: Any | None = None,
    writer_properties: Any | None = None,
    post_commithook_properties: Any | None = None,
    commit_properties: Any | None = None,
) -> Any:
    """Optimize a Delta table by compacting files or applying Z-Order."""
    table = _get_delta_table(table_path)
    try:
        optimizer = table.optimize
    except AttributeError as exc:
        raise RuntimeError("DeltaTable.optimize is not available in this deltalake version.") from exc

    kwargs: dict[str, Any] = {}
    if partition_filters is not None:
        kwargs["partition_filters"] = partition_filters
    if target_size is not None:
        kwargs["target_size"] = target_size
    if max_concurrent_tasks is not None:
        kwargs["max_concurrent_tasks"] = max_concurrent_tasks
    if min_commit_interval is not None:
        kwargs["min_commit_interval"] = min_commit_interval
    if writer_properties is not None:
        kwargs["writer_properties"] = writer_properties
    if post_commithook_properties is not None:
        kwargs["post_commithook_properties"] = post_commithook_properties
    if commit_properties is not None:
        kwargs["commit_properties"] = commit_properties

    if mode == "compact":
        return optimizer.compact(**kwargs)
    if mode == "z_order":
        if not z_order_columns:
            raise ValueError("z_order_columns is required when mode='z_order'.")
        if max_spill_size is not None:
            kwargs["max_spill_size"] = max_spill_size
        return optimizer.z_order(list(z_order_columns), **kwargs)
    raise ValueError(f"Unknown optimize mode: {mode!r}")


def _cleanup_dir(
    directory: Path,
    *,
    keep_ids: set[int],
    threshold: float | None,
    dry_run: bool,
) -> int:
    removed = 0
    if not directory.exists():
        return removed
    for path in directory.glob("*.json"):
        try:
            batch_id = int(path.stem)
        except ValueError:
            continue
        if batch_id in keep_ids:
            continue
        if threshold is not None:
            try:
                mtime = path.stat().st_mtime
            except FileNotFoundError:
                continue
            if mtime >= threshold:
                continue
        if not dry_run:
            path.unlink(missing_ok=True)
        removed += 1
    return removed


def _truncate_dir(
    directory: Path,
    *,
    keep_through_batch_id: int,
    dry_run: bool,
) -> tuple[int, int]:
    removed = 0
    kept = 0
    if not directory.exists():
        return removed, kept
    for path in directory.glob("*.json"):
        try:
            batch_id = int(path.stem)
        except ValueError:
            continue
        if batch_id <= keep_through_batch_id:
            kept += 1
            continue
        if not dry_run:
            path.unlink(missing_ok=True)
        removed += 1
    return removed, kept


def _load_metadata(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _save_metadata(path: Path, payload: dict) -> None:
    atomic_write_json(path, payload)


def _get_delta_table(table_path: str | Path):
    from deltalake import DeltaTable  # type: ignore

    return DeltaTable(str(table_path))
