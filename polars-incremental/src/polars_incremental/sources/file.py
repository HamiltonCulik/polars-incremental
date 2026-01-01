from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import polars as pl

from ..checkpoints.file import FileStreamCheckpoint
from ..checkpoints.types import BatchInfo
from ..utils.options import get_option
from .base import Source


class FileSource(Source):
    def __init__(
        self,
        path: str | Path,
        checkpoint_dir: str | Path,
        source_format: str,
        options: dict[str, Any],
    ) -> None:
        self.path = Path(path)
        self.checkpoint = FileStreamCheckpoint(checkpoint_dir)
        self.file_format = get_option(options, "file_format", default=None)
        if isinstance(self.file_format, str):
            normalized = self.file_format.lower()
            if normalized in {"xlsx", "xls"}:
                self.file_format = "excel"
            elif normalized == "jsonl":
                self.file_format = "ndjson"
        if self.file_format is None:
            normalized_source = source_format.lower()
            if normalized_source in {"csv", "parquet", "excel", "xlsx", "xls", "json", "ndjson", "jsonl", "avro"}:
                if normalized_source in {"excel", "xlsx", "xls"}:
                    self.file_format = "excel"
                elif normalized_source == "jsonl":
                    self.file_format = "ndjson"
                else:
                    self.file_format = normalized_source
            else:
                self.file_format = "parquet"
        default_pattern = "*.parquet"
        if self.file_format == "csv":
            default_pattern = "*.csv"
        elif self.file_format == "excel":
            default_pattern = "*.xlsx"
        elif self.file_format == "json":
            default_pattern = "*.json"
        elif self.file_format == "ndjson":
            default_pattern = "*.ndjson"
        elif self.file_format == "avro":
            default_pattern = "*.avro"
        self.pattern = get_option(options, "pattern", default=default_pattern)
        self.recursive = bool(get_option(options, "recursive", default=False))
        self.max_files = get_option(options, "max_files_per_trigger", default=None)
        self.max_bytes = get_option(options, "max_bytes_per_trigger", default=None)
        self.max_file_age = get_option(options, "max_file_age", default=None)
        self.start_offset = get_option(options, "start_offset", default=None)
        self.start_timestamp = get_option(options, "start_timestamp", default=None)
        self.allow_overwrites = bool(get_option(options, "allow_overwrites", default=False))
        self.clean_source = str(get_option(options, "clean_source", default="off")).lower()
        self.clean_source_archive_dir = get_option(options, "clean_source_archive_dir", default=None)
        if self.clean_source not in {"off", "delete", "archive"}:
            raise ValueError(f"Unsupported clean_source mode: {self.clean_source}")
    def _archive_dir(self) -> Path:
        if self.clean_source_archive_dir is not None:
            return Path(self.clean_source_archive_dir)
        return self.path / "_archive"

    def plan_batch(self) -> BatchInfo | None:
        exclude_dirs: list[Path] | None = None
        if self.clean_source == "archive" and self.recursive:
            archive_dir = self._archive_dir()
            try:
                resolved_archive = archive_dir.resolve()
            except OSError:
                resolved_archive = archive_dir
            try:
                if resolved_archive.relative_to(self.path.resolve()):
                    exclude_dirs = [archive_dir]
            except ValueError:
                exclude_dirs = None
        batch = self.checkpoint.plan_batch(
            input_dir=self.path,
            pattern=self.pattern,
            recursive=self.recursive,
            max_files=self.max_files,
            max_bytes=self.max_bytes,
            max_file_age=self.max_file_age,
            start_offset=self.start_offset,
            start_timestamp=self.start_timestamp,
            allow_overwrites=self.allow_overwrites,
            exclude_dirs=exclude_dirs,
        )
        if batch is None:
            return None
        self.checkpoint.write_offset(batch)
        return batch

    def read_batch(self, batch: BatchInfo) -> pl.DataFrame:
        if not batch.files:
            return pl.DataFrame()
        return self._read_batch_files(batch.files)

    def commit_batch(self, batch: BatchInfo, metadata: dict | None = None) -> None:
        self.checkpoint.commit_batch(batch, metadata=metadata)
        if self.clean_source != "off":
            removed_files = self._clean_source_files(batch.files)
            self.checkpoint._update_file_index([], removed_files=removed_files)

    def _clean_source_files(self, files: list[str]) -> list[str]:
        removed: list[str] = []
        if self.clean_source == "off":
            return removed
        if self.clean_source == "delete":
            for path in files:
                try:
                    Path(path).unlink(missing_ok=True)
                    removed.append(path)
                except OSError:
                    continue
            return removed
        if self.clean_source == "archive":
            archive_dir = self._archive_dir()
            archive_dir.mkdir(parents=True, exist_ok=True)
            for path in files:
                src = Path(path)
                if not src.exists():
                    continue
                dest = archive_dir / src.name
                if dest.exists():
                    stem = dest.stem
                    suffix = dest.suffix
                    counter = 1
                    while True:
                        candidate = archive_dir / f"{stem}_{counter}{suffix}"
                        if not candidate.exists():
                            dest = candidate
                            break
                        counter += 1
                try:
                    shutil.move(str(src), str(dest))
                    removed.append(path)
                except OSError:
                    continue
            return removed
        return removed

    def _read_one(self, path: str) -> pl.DataFrame:
        if self.file_format == "csv":
            return pl.read_csv(path)
        if self.file_format == "excel":
            return pl.read_excel(path)
        if self.file_format == "json":
            return pl.read_json(path)
        if self.file_format == "ndjson":
            return pl.read_ndjson(path)
        if self.file_format == "avro":
            return pl.read_avro(path)
        return pl.read_parquet(path)

    def _read_batch_files(self, files: list[str]) -> pl.DataFrame:
        if not files:
            return pl.DataFrame()
        if self.file_format == "csv":
            return pl.read_csv(files)
        if self.file_format == "parquet":
            return pl.read_parquet(files)
        if self.file_format == "excel":
            frames = [self._read_one(path) for path in files]
            return pl.concat(frames, how="vertical") if frames else pl.DataFrame()
        frames = [self._read_one(path) for path in files]
        return pl.concat(frames, how="vertical") if frames else pl.DataFrame()
