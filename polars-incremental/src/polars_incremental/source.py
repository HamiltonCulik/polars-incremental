from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .sources.base import SourceSpec


class SourceConfig(Protocol):
    def to_spec(self) -> SourceSpec:
        raise NotImplementedError


def _compact_options(options: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in options.items() if value is not None}


@dataclass(frozen=True)
class FilesSource(SourceConfig):
    path: str | Path
    file_format: str | None = None
    pattern: str | None = None
    recursive: bool = False
    max_files_per_trigger: int | None = None
    max_bytes_per_trigger: int | None = None
    max_file_age: float | None = None
    start_offset: str | None = None
    start_timestamp: str | float | None = None
    allow_overwrites: bool = False
    clean_source: str | None = None
    clean_source_archive_dir: str | Path | None = None

    def to_spec(self) -> SourceSpec:
        options = _compact_options(
            {
                "file_format": self.file_format,
                "pattern": self.pattern,
                "recursive": self.recursive,
                "max_files_per_trigger": self.max_files_per_trigger,
                "max_bytes_per_trigger": self.max_bytes_per_trigger,
                "max_file_age": self.max_file_age,
                "start_offset": self.start_offset,
                "start_timestamp": self.start_timestamp,
                "allow_overwrites": self.allow_overwrites,
                "clean_source": self.clean_source,
                "clean_source_archive_dir": self.clean_source_archive_dir,
            }
        )
        return SourceSpec(format="files", path=self.path, options=options)


@dataclass(frozen=True)
class DeltaSource(SourceConfig):
    path: str | Path
    start_offset: str | None = None
    starting_version: int | None = None
    starting_timestamp: str | None = None
    max_files_per_trigger: int | None = 1000
    max_bytes_per_trigger: int | None = None
    ignore_deletes: bool = False
    ignore_changes: bool = False
    read_change_feed: bool = False

    def to_spec(self) -> SourceSpec:
        options = _compact_options(
            {
                "start_offset": self.start_offset,
                "starting_version": self.starting_version,
                "starting_timestamp": self.starting_timestamp,
                "max_files_per_trigger": self.max_files_per_trigger,
                "max_bytes_per_trigger": self.max_bytes_per_trigger,
                "ignore_deletes": self.ignore_deletes,
                "ignore_changes": self.ignore_changes,
                "read_change_feed": self.read_change_feed,
            }
        )
        return SourceSpec(format="delta", path=self.path, options=options)


@dataclass(frozen=True)
class AutoSource(SourceConfig):
    path: str | Path
    file_format: str | None = None
    pattern: str | None = None

    def to_spec(self) -> SourceSpec:
        options = _compact_options(
            {
                "file_format": self.file_format,
                "pattern": self.pattern,
            }
        )
        return SourceSpec(format="auto", path=self.path, options=options)
