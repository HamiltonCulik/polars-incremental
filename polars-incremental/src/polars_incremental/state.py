from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, TypeVar

import polars as pl

T = TypeVar("T")


class JobState:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _fsync_dir(self, path: Path) -> None:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _atomic_write(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
        self._fsync_dir(path.parent)

    def load_json(self, name: str, default: T) -> T:
        path = self.root / f"{name}.json"
        if not path.exists():
            return default
        return json.loads(path.read_text())

    def save_json(self, name: str, obj: Any) -> None:
        path = self.root / f"{name}.json"
        payload = json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")
        self._atomic_write(path, payload)

    def load_parquet(self, name: str) -> pl.DataFrame | None:
        path = self.root / f"{name}.parquet"
        if not path.exists():
            return None
        return pl.read_parquet(path)

    def save_parquet(self, name: str, df: pl.DataFrame) -> None:
        path = self.root / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        df.write_parquet(tmp_path)
        os.replace(tmp_path, path)
        try:
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        except OSError:
            pass
        self._fsync_dir(path.parent)

    def exists(self, name: str, *, kind: str | None = None) -> bool:
        if kind is None:
            return (self.root / f"{name}.json").exists() or (self.root / f"{name}.parquet").exists()
        if kind == "json":
            return (self.root / f"{name}.json").exists()
        if kind == "parquet":
            return (self.root / f"{name}.parquet").exists()
        raise ValueError(f"Unsupported state kind: {kind!r}")

    def delete(self, name: str, *, kind: str | None = None) -> bool:
        removed = False
        if kind is None or kind == "json":
            path = self.root / f"{name}.json"
            if path.exists():
                path.unlink(missing_ok=True)
                removed = True
        if kind is None or kind == "parquet":
            path = self.root / f"{name}.parquet"
            if path.exists():
                path.unlink(missing_ok=True)
                removed = True
        if kind not in (None, "json", "parquet"):
            raise ValueError(f"Unsupported state kind: {kind!r}")
        return removed
