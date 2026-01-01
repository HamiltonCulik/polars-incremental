from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..errors import ChangeDataFeedError
from .types import (
    DeltaBatch,
    DeltaFileEntry,
    DeltaOffset,
    atomic_write_json,
    list_batch_ids,
)

logger = logging.getLogger("polars_incremental")


@dataclass(frozen=True)
class _LogAction:
    index: int
    action_type: str
    path: str | None
    size: int | None
    data_change: bool | None


class DeltaTableCheckpoint:
    """Spark-like checkpointing for Delta table streaming reads (CDF optional)."""

    _SNAPSHOT_CACHE_DIR = "snapshot_cache"
    _SNAPSHOT_DIR = "snapshots"
    _DELTA_DIR = "deltas"
    _SNAPSHOT_EVERY = 100
    _MAX_SNAPSHOTS = 2

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.offset_dir = self.checkpoint_dir / "offsets"
        self.commit_dir = self.checkpoint_dir / "commits"
        self._metadata_path = self.checkpoint_dir / "metadata.json"
        self._ensure_dirs()
        self._metadata = self._load_or_create_metadata()
        self._snapshot_cache: dict[str, Any] | None = self._load_snapshot_cache()

    def _ensure_dirs(self) -> None:
        self.offset_dir.mkdir(parents=True, exist_ok=True)
        self.commit_dir.mkdir(parents=True, exist_ok=True)

    def _load_or_create_metadata(self) -> dict:
        if self._metadata_path.exists():
            return json.loads(self._metadata_path.read_text())
        payload = {
            "format_version": 1,
            "source": "delta",
            "created_at": time.time(),
        }
        atomic_write_json(self._metadata_path, payload)
        return payload

    def _save_metadata(self) -> None:
        atomic_write_json(self._metadata_path, self._metadata)

    def _load_snapshot_cache(self) -> dict[str, Any] | None:
        cache = self._metadata.get("snapshot_cache")
        if not isinstance(cache, dict):
            return None
        version = cache.get("version")
        if not isinstance(version, int):
            return None
        table_id = cache.get("table_id")
        active = cache.get("active")
        if isinstance(active, dict):
            try:
                self._write_snapshot_file(version, dict(active))
                upgraded = {"version": version}
                if table_id is not None:
                    upgraded["table_id"] = table_id
                if self._metadata.get("snapshot_cache") != upgraded:
                    self._metadata["snapshot_cache"] = upgraded
                    self._save_metadata()
                return upgraded
            except OSError:
                return None
        return {"version": version, "table_id": table_id}

    def _store_snapshot_cache(self, cache: dict[str, Any]) -> None:
        self._snapshot_cache = cache
        if self._metadata.get("snapshot_cache") != cache:
            self._metadata["snapshot_cache"] = cache
            self._save_metadata()

    def _clear_snapshot_cache(self) -> None:
        self._snapshot_cache = None
        self._metadata.pop("snapshot_cache", None)
        self._save_metadata()
        cache_root = self._snapshot_cache_root()
        if cache_root.exists():
            for path in cache_root.rglob("*"):
                if path.is_file():
                    path.unlink(missing_ok=True)
            for path in sorted(cache_root.rglob("*"), reverse=True):
                if path.is_dir():
                    path.rmdir()

    def _snapshot_cache_root(self) -> Path:
        return self.checkpoint_dir / self._SNAPSHOT_CACHE_DIR

    def _snapshot_dir(self) -> Path:
        return self._snapshot_cache_root() / self._SNAPSHOT_DIR

    def _delta_dir(self) -> Path:
        return self._snapshot_cache_root() / self._DELTA_DIR

    def _snapshot_path(self, version: int) -> Path:
        return self._snapshot_dir() / f"{version:020d}.json"

    def _delta_path(self, version: int) -> Path:
        return self._delta_dir() / f"{version:020d}.json"

    def _load_snapshot_file(self, version: int) -> dict[str, int | None] | None:
        path = self._snapshot_path(version)
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else None

    def _write_snapshot_file(self, version: int, active: dict[str, int | None]) -> None:
        self._snapshot_dir().mkdir(parents=True, exist_ok=True)
        atomic_write_json(self._snapshot_path(version), active)

    def _load_delta_file(self, version: int) -> tuple[dict[str, int | None], list[str]] | None:
        path = self._delta_path(version)
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            return None
        adds = payload.get("adds")
        removes = payload.get("removes")
        if not isinstance(adds, dict) or not isinstance(removes, list):
            return None
        return dict(adds), [str(path) for path in removes]

    def _write_delta_file(
        self,
        version: int,
        adds: dict[str, int | None],
        removes: list[str],
    ) -> None:
        self._delta_dir().mkdir(parents=True, exist_ok=True)
        payload = {"version": version, "adds": adds, "removes": removes}
        atomic_write_json(self._delta_path(version), payload)

    def _list_snapshot_versions(self) -> list[int]:
        if not self._snapshot_dir().exists():
            return []
        versions: list[int] = []
        for path in self._snapshot_dir().glob("*.json"):
            try:
                versions.append(int(path.stem))
            except ValueError:
                continue
        return sorted(versions)

    def _prune_snapshots(self) -> None:
        versions = self._list_snapshot_versions()
        if len(versions) <= self._MAX_SNAPSHOTS:
            return
        to_delete = versions[: -self._MAX_SNAPSHOTS]
        for version in to_delete:
            self._snapshot_path(version).unlink(missing_ok=True)

    def _prune_deltas(self, up_to_version: int) -> None:
        if not self._delta_dir().exists():
            return
        for path in self._delta_dir().glob("*.json"):
            try:
                version = int(path.stem)
            except ValueError:
                continue
            if version <= up_to_version:
                path.unlink(missing_ok=True)

    def get_schema(self) -> list[dict] | None:
        schema = self._metadata.get("schema")
        return schema if isinstance(schema, list) else None

    def set_schema(self, schema: list[dict]) -> None:
        self._metadata["schema"] = schema
        self._save_metadata()

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

    def read_offset(self, batch_id: int) -> DeltaBatch:
        path = self._batch_path(self.offset_dir, batch_id)
        payload = json.loads(path.read_text())
        offset = DeltaOffset(
            table_id=payload.get("table_id"),
            version=int(payload["version"]),
            index=int(payload["index"]),
            is_initial_snapshot=bool(payload["is_initial_snapshot"]),
        )
        file_entries_payload = payload.get("file_entries")
        file_entries = None
        if isinstance(file_entries_payload, list):
            file_entries = [
                DeltaFileEntry(
                    path=str(entry["path"]),
                    commit_version=int(entry["commit_version"]),
                    commit_timestamp=entry.get("commit_timestamp"),
                    change_type=entry.get("change_type"),
                    size=entry.get("size"),
                )
                for entry in file_entries_payload
            ]
        files = list(payload.get("files", []))
        if not files and file_entries is not None:
            files = [entry.path for entry in file_entries]
        return DeltaBatch(
            batch_id=int(payload["batch_id"]),
            offset=offset,
            files=files,
            created_at=float(payload["created_at"]),
            file_entries=file_entries,
        )

    def write_offset(self, batch: DeltaBatch) -> None:
        path = self._batch_path(self.offset_dir, batch.batch_id)
        if path.exists():
            return
        payload = {
            "batch_id": batch.batch_id,
            "created_at": batch.created_at,
            "table_id": batch.offset.table_id,
            "version": batch.offset.version,
            "index": batch.offset.index,
            "is_initial_snapshot": batch.offset.is_initial_snapshot,
            "files": batch.files,
        }
        if batch.file_entries is not None:
            payload["file_entries"] = [
                {
                    "path": entry.path,
                    "commit_version": entry.commit_version,
                    "commit_timestamp": entry.commit_timestamp,
                    "change_type": entry.change_type,
                    "size": entry.size,
                }
                for entry in batch.file_entries
            ]
        atomic_write_json(path, payload)

    def commit_batch(self, batch: DeltaBatch, metadata: dict | None = None) -> None:
        payload = {
            "batch_id": batch.batch_id,
            "committed_at": time.time(),
            "file_count": len(batch.files),
            "version": batch.offset.version,
            "index": batch.offset.index,
            "is_initial_snapshot": batch.offset.is_initial_snapshot,
            "metadata": metadata or {},
        }
        path = self._batch_path(self.commit_dir, batch.batch_id)
        atomic_write_json(path, payload)

    def _delta_log_dir(self, table_path: str | Path) -> Path:
        return Path(table_path) / "_delta_log"

    def _list_log_versions(self, table_path: str | Path) -> list[int]:
        log_dir = self._delta_log_dir(table_path)
        if not log_dir.exists():
            return []
        versions: list[int] = []
        for path in log_dir.glob("*.json"):
            try:
                versions.append(int(path.stem))
            except ValueError:
                continue
        return sorted(versions)

    def _latest_version(self, table_path: str | Path) -> int | None:
        versions = self._list_log_versions(table_path)
        return versions[-1] if versions else None

    def _warn_if_start_offset_ignored(
        self,
        *,
        start_offset: str | None,
        starting_version: int | None,
        starting_timestamp: str | None,
    ) -> None:
        if start_offset is None and starting_version is None and starting_timestamp is None:
            return
        existing = self._metadata.get("start_offset")
        if not isinstance(existing, dict) or "mode" not in existing:
            return

        desired_mode: str | None = None
        desired_version: int | None = None
        desired_timestamp: str | None = None
        if starting_version is not None:
            desired_mode = "version"
            desired_version = int(starting_version)
        elif starting_timestamp is not None:
            desired_mode = "version"
            desired_timestamp = str(starting_timestamp)
        elif start_offset is not None:
            desired_mode = str(start_offset)

        if desired_mode is None:
            return

        existing_mode = str(existing.get("mode"))
        mismatch = desired_mode != existing_mode
        if desired_mode == "version" and existing_mode == "version":
            if desired_version is not None:
                mismatch = int(existing.get("version", -1)) != desired_version
            elif desired_timestamp is not None:
                mismatch = existing.get("timestamp") != desired_timestamp

        if mismatch:
            logger.warning(
                "start_offset/starting_version/starting_timestamp is ignored because the "
                "checkpoint already stores start_offset=%r. Use a new checkpoint "
                "directory or reset_checkpoint_start_offset(...) to change it.",
                existing,
            )

    def _iter_log_actions(self, table_path: str | Path, version: int) -> Iterator[_LogAction]:
        log_path = self._delta_log_dir(table_path) / f"{version:020d}.json"
        if not log_path.exists():
            return iter(())
        actions: list[_LogAction] = []
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if "add" in payload:
                    add = payload["add"]
                    actions.append(
                        _LogAction(
                            index=len(actions),
                            action_type="add",
                            path=add.get("path"),
                            size=add.get("size"),
                            data_change=add.get("dataChange"),
                        )
                    )
                elif "cdc" in payload:
                    cdc = payload["cdc"]
                    actions.append(
                        _LogAction(
                            index=len(actions),
                            action_type="cdc",
                            path=cdc.get("path"),
                            size=cdc.get("size"),
                            data_change=cdc.get("dataChange"),
                        )
                    )
                elif "remove" in payload:
                    remove = payload["remove"]
                    actions.append(
                        _LogAction(
                            index=len(actions),
                            action_type="remove",
                            path=remove.get("path"),
                            size=None,
                            data_change=remove.get("dataChange"),
                        )
                    )
        return iter(actions)

    def _get_commit_timestamp_ms(self, table_path: str | Path, version: int) -> int | None:
        log_path = self._delta_log_dir(table_path) / f"{version:020d}.json"
        if not log_path.exists():
            return None
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if "commitInfo" in payload:
                    ts = payload["commitInfo"].get("timestamp")
                    if ts is None:
                        return None
                    if isinstance(ts, (int, float)):
                        return int(ts)
        return None

    def _load_table_id(self, table_path: str | Path) -> str | None:
        for version in self._list_log_versions(table_path):
            log_path = self._delta_log_dir(table_path) / f"{version:020d}.json"
            if not log_path.exists():
                continue
            with log_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    if "metaData" in payload:
                        return payload["metaData"].get("id")
        return None

    def _snapshot_state(
        self, table_path: str | Path, version: int
    ) -> list[tuple[str, int | None]]:
        cache = self._snapshot_cache or {}
        cached_version = cache.get("version")
        cached_table_id = cache.get("table_id")
        current_table_id = None
        if cached_table_id is not None:
            current_table_id = self._load_table_id(table_path)
            if current_table_id is not None and cached_table_id != current_table_id:
                self._clear_snapshot_cache()
                cache = {}
                cached_version = None
                cached_table_id = None

        active: dict[str, int | None] | None = None
        if isinstance(cached_version, int):
            active = self._load_snapshot_file(cached_version)
            if active is None:
                self._clear_snapshot_cache()
                cached_version = None
                cached_table_id = None

        if active is None:
            cached_version = None
            active = {}

        def apply_delta(adds: dict[str, int | None], removes: list[str]) -> None:
            for path, size in adds.items():
                active[path] = size
            for path in removes:
                active.pop(path, None)

        start_version = 0 if cached_version is None else cached_version + 1
        for v in range(start_version, version + 1):
            delta = self._load_delta_file(v)
            if delta is None:
                adds: dict[str, int | None] = {}
                removes: list[str] = []
                for action in self._iter_log_actions(table_path, v):
                    if action.path is None:
                        continue
                    if action.action_type == "add":
                        adds[action.path] = action.size
                    elif action.action_type == "remove":
                        removes.append(action.path)
                self._write_delta_file(v, adds, removes)
                apply_delta(adds, removes)
            else:
                adds, removes = delta
                apply_delta(adds, removes)

        if current_table_id is None:
            current_table_id = self._load_table_id(table_path)

        if cached_version is None:
            self._write_snapshot_file(version, active)
            cache_payload = {"version": version}
            if current_table_id is not None:
                cache_payload["table_id"] = current_table_id
            self._store_snapshot_cache(cache_payload)
            self._prune_deltas(up_to_version=version)
            self._prune_snapshots()
        elif version - cached_version >= self._SNAPSHOT_EVERY:
            self._write_snapshot_file(version, active)
            cache_payload = {"version": version}
            if current_table_id is not None:
                cache_payload["table_id"] = current_table_id
            elif cached_table_id is not None:
                cache_payload["table_id"] = cached_table_id
            self._store_snapshot_cache(cache_payload)
            self._prune_deltas(up_to_version=version)
            self._prune_snapshots()

        return [(path, active[path]) for path in sorted(active.keys())]

    def _cdf_entries_for_version(
        self, table_path: str | Path, version: int
    ) -> tuple[list[DeltaFileEntry], int | None, bool]:
        actions = list(self._iter_log_actions(table_path, version))
        commit_ts = self._get_commit_timestamp_ms(table_path, version)
        cdc_actions = [a for a in actions if a.action_type == "cdc" and a.path]
        if cdc_actions:
            entries = [
                DeltaFileEntry(
                    path=str((Path(table_path) / action.path).resolve()),
                    commit_version=version,
                    commit_timestamp=commit_ts,
                    change_type=None,
                    size=action.size,
                )
                for action in cdc_actions
                if action.path is not None
            ]
            return entries, cdc_actions[-1].index, True

        add_actions = [
            a
            for a in actions
            if a.action_type == "add" and a.path and a.data_change is not False
        ]
        remove_actions = [
            a for a in actions if a.action_type == "remove" and a.data_change is not False
        ]
        if remove_actions:
            raise ChangeDataFeedError(
                "Delta change feed required, but commit includes deletes without CDF files."
            )
        if add_actions:
            entries = [
                DeltaFileEntry(
                    path=str((Path(table_path) / action.path).resolve()),
                    commit_version=version,
                    commit_timestamp=commit_ts,
                    change_type="insert",
                    size=action.size,
                )
                for action in add_actions
                if action.path is not None
            ]
            return entries, add_actions[-1].index, False
        return [], actions[-1].index if actions else None, False

    def _parse_starting_timestamp(self, value: str) -> int:
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError:
            dt = datetime.fromisoformat(normalized + "T00:00:00")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    def _resolve_starting_version(
        self, table_path: str | Path, starting_timestamp: str | None
    ) -> int | None:
        versions = self._list_log_versions(table_path)
        if not versions:
            return None
        if starting_timestamp is None:
            return versions[0]
        start_ms = self._parse_starting_timestamp(starting_timestamp)
        for version in versions:
            ts = self._get_commit_timestamp_ms(table_path, version)
            if ts is None:
                continue
            if ts >= start_ms:
                return version
        return None

    def _resolve_start_offset_config(
        self,
        table_path: str | Path,
        start_offset: str | None,
        starting_version: int | None,
        starting_timestamp: str | None,
    ) -> dict | None:
        existing = self._metadata.get("start_offset")
        if isinstance(existing, dict) and "mode" in existing:
            return existing

        if starting_version is not None:
            config = {"mode": "version", "version": int(starting_version)}
            self._metadata["start_offset"] = config
            self._save_metadata()
            return config

        if starting_timestamp is not None:
            resolved = self._resolve_starting_version(table_path, starting_timestamp)
            if resolved is None:
                return None
            config = {
                "mode": "version",
                "version": int(resolved),
                "timestamp": starting_timestamp,
            }
            self._metadata["start_offset"] = config
            self._save_metadata()
            return config

        if start_offset is None or start_offset == "snapshot":
            config = {"mode": "snapshot"}
            self._metadata["start_offset"] = config
            self._save_metadata()
            return config

        if start_offset == "latest":
            latest_version = self._latest_version(table_path)
            if latest_version is None:
                return None
            config = {"mode": "latest", "version": int(latest_version)}
            self._metadata["start_offset"] = config
            self._save_metadata()
            return config

        if start_offset == "earliest":
            config = {"mode": "version", "version": 0}
            self._metadata["start_offset"] = config
            self._save_metadata()
            return config

        raise ValueError(f"Unsupported start_offset mode: {start_offset}")

    def _next_batch_id(self) -> int:
        latest_offset = self.latest_offset_batch_id()
        return 0 if latest_offset is None else latest_offset + 1

    def _build_snapshot_batch(
        self,
        table_path: str | Path,
        version: int,
        start_index: int,
        max_files: int | None,
        max_bytes: int | None,
        with_entries: bool = False,
        change_type: str | None = None,
    ) -> DeltaBatch | None:
        entries = self._snapshot_state(table_path, version)
        if start_index >= len(entries):
            return None
        files: list[str] = []
        selected_entries: list[tuple[str, int | None]] = []
        total_bytes = 0
        index = start_index
        while index < len(entries):
            path, size = entries[index]
            size_value = int(size or 0)
            next_total = total_bytes + size_value
            if files and max_files is not None and len(files) >= max_files:
                break
            if files and max_bytes is not None and next_total > max_bytes:
                break
            files.append(str((Path(table_path) / path).resolve()))
            selected_entries.append((path, size))
            total_bytes = next_total
            index += 1
        if not files and start_index < len(entries):
            path, size = entries[start_index]
            files.append(str((Path(table_path) / path).resolve()))
            selected_entries.append((path, size))
            index = start_index + 1
        if not files:
            return None
        file_entries = None
        if with_entries:
            commit_ts = self._get_commit_timestamp_ms(table_path, version)
            file_entries = [
                DeltaFileEntry(
                    path=str((Path(table_path) / path).resolve()),
                    commit_version=version,
                    commit_timestamp=commit_ts,
                    change_type=change_type,
                    size=size,
                )
                for path, size in selected_entries
            ]
        offset = DeltaOffset(
            table_id=self._load_table_id(table_path),
            version=version,
            index=index - 1,
            is_initial_snapshot=True,
        )
        return DeltaBatch(
            batch_id=self._next_batch_id(),
            offset=offset,
            files=files,
            created_at=time.time(),
            file_entries=file_entries,
        )

    def _build_log_batch(
        self,
        table_path: str | Path,
        start_version: int,
        start_index: int,
        max_files: int | None,
        max_bytes: int | None,
        ignore_deletes: bool,
        ignore_changes: bool,
    ) -> DeltaBatch | None:
        versions = self._list_log_versions(table_path)
        if not versions:
            return None
        latest_version = versions[-1]
        if start_version > latest_version:
            return None

        files: list[str] = []
        total_bytes = 0
        end_version: int | None = None
        end_index: int | None = None
        advanced_without_files = False

        for version in range(start_version, latest_version + 1):
            actions = list(self._iter_log_actions(table_path, version))
            has_add = any(
                action.action_type == "add" and action.data_change is not False
                for action in actions
            )
            has_remove = any(
                action.action_type == "remove" and action.data_change is not False
                for action in actions
            )

            if has_remove and not ignore_changes:
                if ignore_deletes and not has_add:
                    advanced_without_files = True
                else:
                    raise RuntimeError(
                        "Delta source saw data changes; set ignore_changes to true to allow."
                    )
            if has_remove and ignore_changes and not has_add:
                advanced_without_files = True

            for action in actions:
                if version == start_version and action.index <= start_index:
                    continue
                if action.action_type == "remove":
                    if action.data_change is False:
                        end_version = version
                        end_index = action.index
                        advanced_without_files = True
                        continue
                    end_version = version
                    end_index = action.index
                    continue
                if action.action_type != "add" or action.path is None:
                    continue
                if action.data_change is False:
                    end_version = version
                    end_index = action.index
                    advanced_without_files = True
                    continue

                size_value = int(action.size or 0)
                next_total = total_bytes + size_value
                if files and max_files is not None and len(files) >= max_files:
                    return self._finalize_log_batch(table_path, files, end_version, end_index)
                if files and max_bytes is not None and next_total > max_bytes:
                    return self._finalize_log_batch(table_path, files, end_version, end_index)
                files.append(str((Path(table_path) / action.path).resolve()))
                total_bytes = next_total
                end_version = version
                end_index = action.index

        if files:
            return self._finalize_log_batch(table_path, files, end_version, end_index)
        if advanced_without_files and end_version is not None and end_index is not None:
            offset = DeltaOffset(
                table_id=self._load_table_id(table_path),
                version=end_version,
                index=end_index,
                is_initial_snapshot=False,
            )
            return DeltaBatch(
                batch_id=self._next_batch_id(),
                offset=offset,
                files=[],
                created_at=time.time(),
            )
        return None

    def _finalize_log_batch(
        self,
        table_path: str | Path,
        files: list[str],
        end_version: int | None,
        end_index: int | None,
    ) -> DeltaBatch | None:
        if end_version is None or end_index is None:
            return None
        offset = DeltaOffset(
            table_id=self._load_table_id(table_path),
            version=end_version,
            index=end_index,
            is_initial_snapshot=False,
        )
        return DeltaBatch(
            batch_id=self._next_batch_id(),
            offset=offset,
            files=files,
            created_at=time.time(),
        )

    def _build_cdf_log_batch(
        self,
        table_path: str | Path,
        start_version: int,
        start_index: int,
        max_files: int | None,
        max_bytes: int | None,
    ) -> DeltaBatch | None:
        versions = self._list_log_versions(table_path)
        if not versions:
            return None
        if start_index >= 0:
            start_version += 1
        latest_version = versions[-1]
        if start_version > latest_version:
            return None

        file_entries: list[DeltaFileEntry] = []
        total_files = 0
        total_bytes = 0
        end_version: int | None = None
        end_index: int | None = None
        advanced_without_files = False

        for version in range(start_version, latest_version + 1):
            entries, last_index, _ = self._cdf_entries_for_version(table_path, version)
            if not entries:
                advanced_without_files = True
                end_version = version
                end_index = 0 if last_index is None else last_index
                continue

            entries_size = sum(int(entry.size or 0) for entry in entries)
            would_exceed_files = (
                max_files is not None
                and file_entries
                and total_files + len(entries) > max_files
            )
            would_exceed_bytes = (
                max_bytes is not None
                and file_entries
                and total_bytes + entries_size > max_bytes
            )
            if would_exceed_files or would_exceed_bytes:
                break

            file_entries.extend(entries)
            total_files += len(entries)
            total_bytes += entries_size
            end_version = version
            end_index = 0 if last_index is None else last_index

            if max_files is not None and total_files >= max_files:
                break
            if max_bytes is not None and total_bytes >= max_bytes:
                break

        if file_entries:
            offset = DeltaOffset(
                table_id=self._load_table_id(table_path),
                version=end_version or start_version,
                index=end_index or 0,
                is_initial_snapshot=False,
            )
            return DeltaBatch(
                batch_id=self._next_batch_id(),
                offset=offset,
                files=[entry.path for entry in file_entries],
                created_at=time.time(),
                file_entries=file_entries,
            )

        if advanced_without_files and end_version is not None and end_index is not None:
            offset = DeltaOffset(
                table_id=self._load_table_id(table_path),
                version=end_version,
                index=end_index,
                is_initial_snapshot=False,
            )
            return DeltaBatch(
                batch_id=self._next_batch_id(),
                offset=offset,
                files=[],
                created_at=time.time(),
            )
        return None

    def plan_batch(
        self,
        table_path: str | Path,
        *,
        start_offset: str | None = None,
        starting_version: int | None = None,
        starting_timestamp: str | None = None,
        max_files: int | None = 1000,
        max_bytes: int | None = None,
        ignore_deletes: bool = False,
        ignore_changes: bool = False,
        read_change_feed: bool = False,
    ) -> DeltaBatch | None:
        self._warn_if_start_offset_ignored(
            start_offset=start_offset,
            starting_version=starting_version,
            starting_timestamp=starting_timestamp,
        )
        latest_offset = self.latest_offset_batch_id()
        latest_commit = self.latest_commit_batch_id()
        stored_batch = None
        if latest_commit is not None:
            stored_batch = self.read_offset(latest_commit)
        elif latest_offset is not None:
            stored_batch = self.read_offset(latest_offset)
        if stored_batch is not None:
            stored_table_id = stored_batch.offset.table_id
            current_table_id = self._load_table_id(table_path)
            if (
                stored_table_id is not None
                and current_table_id is not None
                and stored_table_id != current_table_id
            ):
                raise RuntimeError(
                    "Delta table id changed for this checkpoint; "
                    "use a new checkpoint directory or reset offsets."
                )
        if latest_offset is not None and (
            latest_commit is None or latest_offset > latest_commit
        ):
            return self.read_offset(latest_offset)

        last_committed = None
        if latest_commit is not None:
            last_committed = self.read_offset(latest_commit)

        if last_committed is None:
            config = self._resolve_start_offset_config(
                table_path=table_path,
                start_offset=start_offset,
                starting_version=starting_version,
                starting_timestamp=starting_timestamp,
            )
            if config is None:
                return None
            mode = config["mode"]
            if mode == "snapshot":
                latest_version = self._latest_version(table_path)
                if latest_version is None:
                    return None
                return self._build_snapshot_batch(
                    table_path=table_path,
                    version=latest_version,
                    start_index=0,
                    max_files=max_files,
                    max_bytes=max_bytes,
                    with_entries=read_change_feed,
                    change_type="insert" if read_change_feed else None,
                )

            if mode == "latest":
                latest_version = int(config["version"])
                start_version = latest_version + 1
            else:
                start_version = int(config["version"])

            if read_change_feed:
                return self._build_cdf_log_batch(
                    table_path=table_path,
                    start_version=start_version,
                    start_index=-1,
                    max_files=max_files,
                    max_bytes=max_bytes,
                )
            return self._build_log_batch(
                table_path=table_path,
                start_version=start_version,
                start_index=-1,
                max_files=max_files,
                max_bytes=max_bytes,
                ignore_deletes=ignore_deletes,
                ignore_changes=ignore_changes,
            )

        if last_committed.offset.is_initial_snapshot:
            snapshot_version = last_committed.offset.version
            entries = self._snapshot_state(table_path, snapshot_version)
            next_index = last_committed.offset.index + 1
            if next_index < len(entries):
                return self._build_snapshot_batch(
                    table_path=table_path,
                    version=snapshot_version,
                    start_index=next_index,
                    max_files=max_files,
                    max_bytes=max_bytes,
                    with_entries=read_change_feed,
                    change_type="insert" if read_change_feed else None,
                )
            if read_change_feed:
                return self._build_cdf_log_batch(
                    table_path=table_path,
                    start_version=snapshot_version + 1,
                    start_index=-1,
                    max_files=max_files,
                    max_bytes=max_bytes,
                )
            return self._build_log_batch(
                table_path=table_path,
                start_version=snapshot_version + 1,
                start_index=-1,
                max_files=max_files,
                max_bytes=max_bytes,
                ignore_deletes=ignore_deletes,
                ignore_changes=ignore_changes,
            )

        if read_change_feed:
            return self._build_cdf_log_batch(
                table_path=table_path,
                start_version=last_committed.offset.version,
                start_index=last_committed.offset.index,
                max_files=max_files,
                max_bytes=max_bytes,
            )
        return self._build_log_batch(
            table_path=table_path,
            start_version=last_committed.offset.version,
            start_index=last_committed.offset.index,
            max_files=max_files,
            max_bytes=max_bytes,
            ignore_deletes=ignore_deletes,
            ignore_changes=ignore_changes,
        )
