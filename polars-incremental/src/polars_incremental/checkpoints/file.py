from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .types import BatchInfo, atomic_write_json, list_batch_ids

logger = logging.getLogger("polars_incremental")


class FileStreamCheckpoint:
    """Lightweight checkpointing for file-based micro-batches."""

    _INDEX_DIRNAME = "file_index"
    _INDEX_FORMAT = "sharded-v1"
    _INDEX_SHARDS = 256

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.offset_dir = self.checkpoint_dir / "offsets"
        self.commit_dir = self.checkpoint_dir / "commits"
        self._metadata_path = self.checkpoint_dir / "metadata.json"
        self._ensure_dirs()
        self._metadata = self._load_or_create_metadata()
        self._migrate_legacy_index()

    def _ensure_dirs(self) -> None:
        self.offset_dir.mkdir(parents=True, exist_ok=True)
        self.commit_dir.mkdir(parents=True, exist_ok=True)

    def _load_or_create_metadata(self) -> dict:
        if self._metadata_path.exists():
            return json.loads(self._metadata_path.read_text())
        payload = {
            "format_version": 1,
            "created_at": time.time(),
        }
        atomic_write_json(self._metadata_path, payload)
        return payload

    def _save_metadata(self) -> None:
        atomic_write_json(self._metadata_path, self._metadata)

    def _index_dir(self) -> Path:
        return self.checkpoint_dir / self._INDEX_DIRNAME

    def _index_shard_id(self, path: str) -> str:
        digest = hashlib.md5(path.encode("utf-8")).hexdigest()
        return digest[:2]

    def _shard_path(self, shard_id: str) -> Path:
        return self._index_dir() / f"{shard_id}.json"

    def _read_shard(self, shard_id: str) -> dict:
        path = self._shard_path(shard_id)
        if not path.exists():
            return {}
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}

    def _write_shard(self, shard_id: str, payload: dict) -> None:
        path = self._shard_path(shard_id)
        atomic_write_json(path, payload)

    def _ensure_index_metadata(self) -> None:
        changed = False
        if self._metadata.get("file_index_format") != self._INDEX_FORMAT:
            self._metadata["file_index_format"] = self._INDEX_FORMAT
            changed = True
        if self._metadata.get("file_index_shards") != self._INDEX_SHARDS:
            self._metadata["file_index_shards"] = self._INDEX_SHARDS
            changed = True
        if changed:
            self._save_metadata()

    def _migrate_legacy_index(self) -> None:
        legacy = self._metadata.get("file_index")
        if not isinstance(legacy, dict):
            return
        self._index_dir().mkdir(parents=True, exist_ok=True)
        shards: dict[str, dict] = {}
        for path, sig in legacy.items():
            shard_id = self._index_shard_id(str(path))
            shard = shards.get(shard_id)
            if shard is None:
                shard = {}
                shards[shard_id] = shard
            shard[str(path)] = sig
        for shard_id, payload in shards.items():
            self._write_shard(shard_id, payload)
        self._metadata.pop("file_index", None)
        self._ensure_index_metadata()

    def get_schema(self) -> list[dict] | None:
        schema = self._metadata.get("schema")
        return schema if isinstance(schema, list) else None

    def set_schema(self, schema: list[dict]) -> None:
        self._metadata["schema"] = schema
        self._save_metadata()

    def _file_signature(self, path: str) -> dict | None:
        try:
            stat = Path(path).stat()
        except FileNotFoundError:
            return None
        return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}

    def _file_index(self) -> dict[str, dict]:
        if isinstance(self._metadata.get("file_index"), dict):
            self._migrate_legacy_index()
        if self._metadata.get("file_index_format") == self._INDEX_FORMAT:
            index_dir = self._index_dir()
            if not index_dir.exists():
                return {}
            merged: dict[str, dict] = {}
            for path in index_dir.glob("*.json"):
                payload = json.loads(path.read_text())
                if isinstance(payload, dict):
                    merged.update(payload)
            return merged
        return {}

    def _update_file_index(self, files: list[str], removed_files: list[str] | None = None) -> None:
        if isinstance(self._metadata.get("file_index"), dict):
            self._migrate_legacy_index()
        self._index_dir().mkdir(parents=True, exist_ok=True)
        self._ensure_index_metadata()
        changed_shards: set[str] = set()
        shard_payloads: dict[str, dict] = {}

        for path in files:
            sig = self._file_signature(path)
            if sig is None:
                continue
            shard_id = self._index_shard_id(path)
            payload = shard_payloads.get(shard_id)
            if payload is None:
                payload = self._read_shard(shard_id)
                shard_payloads[shard_id] = payload
            payload[path] = sig
            changed_shards.add(shard_id)

        if removed_files:
            for path in removed_files:
                shard_id = self._index_shard_id(path)
                payload = shard_payloads.get(shard_id)
                if payload is None:
                    payload = self._read_shard(shard_id)
                    shard_payloads[shard_id] = payload
                if path in payload:
                    payload.pop(path, None)
                    changed_shards.add(shard_id)

        for shard_id in changed_shards:
            payload = shard_payloads.get(shard_id) or {}
            self._write_shard(shard_id, payload)

    def _batch_path(self, directory: Path, batch_id: int) -> Path:
        return directory / f"{batch_id}.json"

    def latest_offset_batch_id(self) -> int | None:
        ids = list_batch_ids(self.offset_dir)
        return ids[-1] if ids else None

    def latest_commit_batch_id(self) -> int | None:
        ids = list_batch_ids(self.commit_dir)
        return ids[-1] if ids else None

    def committed_batch_ids(self) -> list[int]:
        return list_batch_ids(self.commit_dir)

    def read_offset(self, batch_id: int) -> BatchInfo:
        path = self._batch_path(self.offset_dir, batch_id)
        payload = json.loads(path.read_text())
        return BatchInfo(
            batch_id=payload["batch_id"],
            files=list(payload["files"]),
            created_at=payload["created_at"],
        )

    def _load_committed_files(self) -> set[str]:
        index = self._file_index()
        if index:
            return set(index.keys())
        committed_files: set[str] = set()
        for batch_id in self.committed_batch_ids():
            try:
                batch = self.read_offset(batch_id)
            except FileNotFoundError:
                continue
            committed_files.update(batch.files)
        return committed_files

    def _list_input_files(
        self,
        input_dir: str | Path,
        pattern: str,
        recursive: bool,
        exclude_dirs: list[str | Path] | None = None,
    ) -> list[str]:
        base = Path(input_dir)
        normalized_excludes: list[Path] = []
        if exclude_dirs:
            for path in exclude_dirs:
                try:
                    normalized_excludes.append(Path(path).resolve())
                except OSError:
                    normalized_excludes.append(Path(path))

        def is_excluded(path: Path) -> bool:
            if not normalized_excludes:
                return False
            resolved = path
            for excluded in normalized_excludes:
                try:
                    resolved.relative_to(excluded)
                    return True
                except ValueError:
                    continue
            return False

        if recursive:
            glob_iter = base.rglob(pattern)
        else:
            glob_iter = base.glob(pattern)
        files: list[Path] = []
        for path in glob_iter:
            if not path.is_file():
                continue
            resolved = path.resolve()
            if is_excluded(resolved):
                continue
            files.append(resolved)
        files_sorted = sorted(files, key=lambda p: str(p))
        return [str(path) for path in files_sorted]

    def _coerce_timestamp(self, value: float | str) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError:
            dt = datetime.fromisoformat(normalized + "T00:00:00")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()

    def _resolve_start_offset(
        self,
        start_offset: str | None,
        start_timestamp: float | str | None,
    ) -> tuple[str, float | None]:
        existing = self._metadata.get("start_offset")
        if isinstance(existing, dict) and "mode" in existing:
            return str(existing["mode"]), existing.get("timestamp")

        mode = start_offset or "earliest"
        ts: float | None = None
        if start_timestamp is not None:
            mode = "timestamp"
            ts = self._coerce_timestamp(start_timestamp)
        elif mode == "latest":
            ts = time.time()
        elif mode not in ("earliest", "latest"):
            raise ValueError(f"Unsupported start_offset mode: {mode}")

        self._metadata["start_offset"] = {"mode": mode, "timestamp": ts}
        self._save_metadata()
        return mode, ts

    def _filter_by_timestamp(self, files: list[str], start_ts: float) -> list[str]:
        filtered: list[str] = []
        for path in files:
            try:
                stat = Path(path).stat()
            except FileNotFoundError:
                continue
            if stat.st_mtime >= start_ts:
                filtered.append(path)
        return filtered

    def _filter_by_max_age(
        self,
        files: list[str],
        max_age_seconds: float,
        reference_ts: float | None = None,
    ) -> list[str]:
        if max_age_seconds is None or max_age_seconds < 0:
            return files
        if reference_ts is None:
            reference_ts = self._latest_mtime(files)
        if reference_ts is None:
            return []
        threshold = reference_ts - max_age_seconds
        filtered: list[str] = []
        for path in files:
            try:
                stat = Path(path).stat()
            except FileNotFoundError:
                continue
            if stat.st_mtime >= threshold:
                filtered.append(path)
        return filtered

    def _latest_mtime(self, files: list[str]) -> float | None:
        latest: float | None = None
        for path in files:
            try:
                stat = Path(path).stat()
            except FileNotFoundError:
                continue
            mtime = stat.st_mtime
            if latest is None or mtime > latest:
                latest = mtime
        return latest

    def _prune_index_by_age(
        self,
        max_age_seconds: float,
        reference_ts: float | None = None,
    ) -> None:
        if max_age_seconds is None or max_age_seconds < 0:
            return
        if reference_ts is None:
            return
        threshold_ns = int((reference_ts - max_age_seconds) * 1e9)
        index_dir = self._index_dir()
        if not index_dir.exists():
            return
        changed_shards: set[str] = set()
        for shard_path in index_dir.glob("*.json"):
            shard_id = shard_path.stem
            payload = self._read_shard(shard_id)
            removed = False
            for path, sig in list(payload.items()):
                mtime_ns = sig.get("mtime_ns") if isinstance(sig, dict) else None
                if isinstance(mtime_ns, int) and mtime_ns < threshold_ns:
                    payload.pop(path, None)
                    removed = True
            if removed:
                changed_shards.add(shard_id)
                self._write_shard(shard_id, payload)

    def _recover_or_plan_batch(
        self,
        input_dir: str | Path,
        pattern: str,
        recursive: bool,
        max_files: int | None,
        max_bytes: int | None,
        start_offset: str | None,
        start_timestamp: float | None,
        allow_overwrites: bool,
        max_file_age: float | None,
        exclude_dirs: list[str | Path] | None,
    ) -> BatchInfo | None:
        latest_offset = self.latest_offset_batch_id()
        latest_commit = self.latest_commit_batch_id()
        if latest_offset is not None and (
            latest_commit is None or latest_offset > latest_commit
        ):
            return self.read_offset(latest_offset)

        committed_files = self._load_committed_files()
        candidates = self._list_input_files(
            input_dir,
            pattern,
            recursive,
            exclude_dirs=exclude_dirs,
        )
        reference_ts: float | None = None
        if max_file_age is not None:
            reference_ts = self._latest_mtime(candidates)
            if reference_ts is not None:
                self._prune_index_by_age(max_file_age, reference_ts=reference_ts)
        if max_file_age is not None:
            candidates = self._filter_by_max_age(
                candidates,
                max_file_age,
                reference_ts=reference_ts,
            )
        if allow_overwrites:
            index = self._file_index()
            missing_committed: list[str] = []
            new_files: list[str] = []
            for path in candidates:
                sig = self._file_signature(path)
                if sig is None:
                    continue
                entry = index.get(path)
                if entry is None:
                    if path in committed_files:
                        missing_committed.append(path)
                        continue
                    new_files.append(path)
                    continue
                if entry.get("mtime_ns") != sig.get("mtime_ns") or entry.get("size") != sig.get("size"):
                    new_files.append(path)
            if missing_committed:
                self._update_file_index(missing_committed)
        else:
            new_files = [path for path in candidates if path not in committed_files]
        if start_offset in ("latest", "timestamp") and start_timestamp is not None:
            new_files = self._filter_by_timestamp(new_files, start_timestamp)
        if max_files is not None or max_bytes is not None:
            selected: list[str] = []
            total_bytes = 0
            for path in new_files:
                if max_files is not None and len(selected) >= max_files:
                    break
                size = 0
                if max_bytes is not None:
                    try:
                        size = Path(path).stat().st_size
                    except FileNotFoundError:
                        continue
                    if selected and total_bytes + size > max_bytes:
                        break
                selected.append(path)
                total_bytes += size
            new_files = selected

        if not new_files:
            return None

        next_batch_id = 0 if latest_offset is None else latest_offset + 1
        return BatchInfo(
            batch_id=next_batch_id,
            files=new_files,
            created_at=time.time(),
        )

    def plan_batch(
        self,
        input_dir: str | Path,
        pattern: str = "*.parquet",
        recursive: bool = False,
        max_files: int | None = None,
        max_bytes: int | None = None,
        start_offset: str | None = None,
        start_timestamp: float | str | None = None,
        allow_overwrites: bool = False,
        max_file_age: float | None = None,
        exclude_dirs: list[str | Path] | None = None,
    ) -> BatchInfo | None:
        self._warn_if_start_offset_ignored(start_offset, start_timestamp)
        latest_offset = self.latest_offset_batch_id()
        latest_commit = self.latest_commit_batch_id()
        resolved_offset = None
        resolved_timestamp = None
        if latest_offset is None and latest_commit is None:
            resolved_offset, resolved_timestamp = self._resolve_start_offset(
                start_offset=start_offset,
                start_timestamp=start_timestamp,
            )
        return self._recover_or_plan_batch(
            input_dir=input_dir,
            pattern=pattern,
            recursive=recursive,
            max_files=max_files,
            max_bytes=max_bytes,
            start_offset=resolved_offset,
            start_timestamp=resolved_timestamp,
            allow_overwrites=allow_overwrites,
            max_file_age=max_file_age,
            exclude_dirs=exclude_dirs,
        )

    def _warn_if_start_offset_ignored(
        self,
        start_offset: str | None,
        start_timestamp: float | str | None,
    ) -> None:
        if start_offset is None and start_timestamp is None:
            return
        existing = self._metadata.get("start_offset")
        if not isinstance(existing, dict) or "mode" not in existing:
            return

        desired_mode: str | None = None
        desired_ts: float | None = None
        if start_timestamp is not None:
            desired_mode = "timestamp"
            try:
                desired_ts = self._coerce_timestamp(start_timestamp)
            except ValueError:
                desired_ts = None
        elif start_offset is not None:
            desired_mode = str(start_offset)

        if desired_mode is None:
            return

        existing_mode = str(existing.get("mode"))
        mismatch = desired_mode != existing_mode
        if desired_mode == "timestamp" and existing_mode == "timestamp":
            existing_ts = existing.get("timestamp")
            if isinstance(existing_ts, (int, float)) and desired_ts is not None:
                mismatch = abs(float(existing_ts) - desired_ts) > 1e-6
            else:
                mismatch = desired_ts is not None

        if mismatch:
            logger.warning(
                "start_offset/start_timestamp is ignored because the checkpoint already "
                "stores start_offset=%r. Use a new checkpoint directory or "
                "reset_checkpoint_start_offset(...) to change it.",
                existing,
            )

    def write_offset(self, batch: BatchInfo) -> None:
        path = self._batch_path(self.offset_dir, batch.batch_id)
        if path.exists():
            return
        payload = {
            "batch_id": batch.batch_id,
            "created_at": batch.created_at,
            "files": batch.files,
        }
        atomic_write_json(path, payload)

    def commit_batch(self, batch: BatchInfo, metadata: dict | None = None) -> None:
        payload = {
            "batch_id": batch.batch_id,
            "committed_at": time.time(),
            "file_count": len(batch.files),
            "metadata": metadata or {},
        }
        path = self._batch_path(self.commit_dir, batch.batch_id)
        atomic_write_json(path, payload)
        self._update_file_index(batch.files)

    def list_pending_batches(self) -> list[BatchInfo]:
        latest_commit = self.latest_commit_batch_id()
        latest_offset = self.latest_offset_batch_id()
        if latest_offset is None:
            return []
        pending: list[BatchInfo] = []
        start = 0 if latest_commit is None else latest_commit + 1
        for batch_id in range(start, latest_offset + 1):
            try:
                pending.append(self.read_offset(batch_id))
            except FileNotFoundError:
                continue
        return pending


def iter_new_files(
    input_dir: str | Path,
    checkpoint_dir: str | Path,
    pattern: str = "*.parquet",
    recursive: bool = False,
    start_offset: str | None = None,
    start_timestamp: float | str | None = None,
    allow_overwrites: bool = False,
    max_bytes: int | None = None,
    max_file_age: float | None = None,
) -> Iterable[str]:
    checkpoint = FileStreamCheckpoint(checkpoint_dir)
    batch = checkpoint.plan_batch(
        input_dir=input_dir,
        pattern=pattern,
        recursive=recursive,
        start_offset=start_offset,
        start_timestamp=start_timestamp,
        allow_overwrites=allow_overwrites,
        max_bytes=max_bytes,
        max_file_age=max_file_age,
    )
    if batch is None:
        return []
    checkpoint.write_offset(batch)
    return list(batch.files)
