from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from ..errors import UnsupportedFormatError
from ..utils.options import get_option


@dataclass(frozen=True)
class SourceSpec:
    format: str
    path: str | Path
    options: dict[str, Any]

    def with_checkpoint(self, checkpoint_dir: str | Path) -> "Source":
        from .delta import DeltaSource
        from .file import FileSource

        fmt = self.format.lower()
        if fmt in {"xlsx", "xls"}:
            fmt = "excel"
        elif fmt == "jsonl":
            fmt = "ndjson"
        if fmt == "auto":
            fmt = _infer_source_format(self.path, self.options)
        if fmt in {"delta"}:
            return DeltaSource(self.path, checkpoint_dir, self.options)
        if fmt in {"file", "files", "parquet", "csv", "excel", "json", "ndjson", "avro"}:
            return FileSource(self.path, checkpoint_dir, fmt, self.options)
        raise UnsupportedFormatError(f"Unsupported source format: {self.format}")


class Source:
    def plan_batch(self) -> Any:
        raise NotImplementedError

    def read_batch(self, batch: Any) -> pl.DataFrame:
        raise NotImplementedError

    def commit_batch(self, batch: Any, metadata: dict | None = None) -> None:
        raise NotImplementedError


def _infer_source_format(path: str | Path, options: dict[str, Any]) -> str:
    file_format = get_option(options, "file_format", default=None)
    normalized_file_format = _normalize_file_format(file_format)
    if normalized_file_format is not None:
        return normalized_file_format

    pattern = get_option(options, "pattern", default=None)
    inferred = _infer_format_from_pattern(pattern)
    if inferred is not None:
        return inferred

    base = Path(path)
    try:
        delta_log = base / "_delta_log"
        if delta_log.exists() and delta_log.is_dir():
            return "delta"
    except OSError:
        pass

    try:
        if base.is_file():
            inferred = _infer_format_from_pattern(base.name)
            if inferred is not None:
                return inferred
    except OSError:
        pass

    return "parquet"


def _infer_format_from_pattern(pattern: Any) -> str | None:
    if not isinstance(pattern, str):
        return None
    suffixes = [suffix.lower() for suffix in Path(pattern).suffixes]
    if ".csv" in suffixes:
        return "csv"
    if ".jsonl" in suffixes or ".ndjson" in suffixes:
        return "ndjson"
    if ".json" in suffixes:
        return "json"
    if ".avro" in suffixes:
        return "avro"
    if ".xlsx" in suffixes or ".xls" in suffixes:
        return "excel"
    if ".parquet" in suffixes or ".parq" in suffixes:
        return "parquet"
    return None


def _normalize_file_format(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"csv", "parquet", "json", "ndjson", "avro"}:
        return normalized
    if normalized in {"excel", "xlsx", "xls"}:
        return "excel"
    if normalized == "jsonl":
        return "ndjson"
    return None
