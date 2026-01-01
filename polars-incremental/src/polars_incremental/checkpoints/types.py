from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BatchInfo:
    batch_id: int
    files: list[str]
    created_at: float


@dataclass(frozen=True)
class DeltaOffset:
    table_id: str | None
    version: int
    index: int
    is_initial_snapshot: bool


@dataclass(frozen=True)
class DeltaFileEntry:
    path: str
    commit_version: int
    commit_timestamp: int | None
    change_type: str | None
    size: int | None


@dataclass(frozen=True)
class DeltaBatch:
    batch_id: int
    offset: DeltaOffset
    files: list[str]
    created_at: float
    file_entries: list[DeltaFileEntry] | None = None


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)
    _fsync_dir(path.parent)


def list_batch_ids(directory: Path) -> list[int]:
    ids: list[int] = []
    for path in directory.glob("*.json"):
        try:
            ids.append(int(path.stem))
        except ValueError:
            continue
    return sorted(ids)
