from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

import tomllib

from .errors import UnsupportedFormatError
from .schema import SchemaEvolution
from .source import AutoSource, DeltaSource, FilesSource, SourceConfig

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    format: str
    path: str | Path
    options: dict[str, Any] = field(default_factory=dict)

    def to_source(self) -> SourceConfig:
        fmt = self.format.lower()
        options = dict(self.options)
        if fmt == "auto":
            return AutoSource(
                path=self.path,
                file_format=options.get("file_format"),
                pattern=options.get("pattern"),
            )
        if fmt in {"files", "file", "parquet", "csv", "excel", "json", "ndjson", "avro"}:
            file_format = options.get("file_format")
            if fmt not in {"files", "file"}:
                file_format = fmt
            return FilesSource(
                path=self.path,
                file_format=file_format,
                pattern=options.get("pattern"),
                recursive=bool(options.get("recursive", False)),
                max_files_per_trigger=options.get("max_files_per_trigger"),
                max_bytes_per_trigger=options.get("max_bytes_per_trigger"),
                max_file_age=options.get("max_file_age"),
                start_offset=options.get("start_offset"),
                start_timestamp=options.get("start_timestamp"),
                allow_overwrites=bool(options.get("allow_overwrites", False)),
                clean_source=options.get("clean_source"),
                clean_source_archive_dir=options.get("clean_source_archive_dir"),
            )
        if fmt == "delta":
            return DeltaSource(
                path=self.path,
                start_offset=options.get("start_offset"),
                starting_version=options.get("starting_version"),
                starting_timestamp=options.get("starting_timestamp"),
                max_files_per_trigger=options.get("max_files_per_trigger", 1000),
                max_bytes_per_trigger=options.get("max_bytes_per_trigger"),
                ignore_deletes=bool(options.get("ignore_deletes", False)),
                ignore_changes=bool(options.get("ignore_changes", False)),
                read_change_feed=bool(options.get("read_change_feed", False)),
            )
        raise UnsupportedFormatError(f"Unsupported source format: {self.format}")

    def to_schema_evolution(self) -> SchemaEvolution | None:
        schema_evolution, _ = SchemaEvolution.from_options(dict(self.options))
        return schema_evolution


class Catalog(Protocol):
    def resolve(self, name: str) -> DatasetSpec:
        raise NotImplementedError

    def get_source(self, name: str) -> SourceConfig:
        raise NotImplementedError


class LocalCatalog:
    def __init__(self, source: str | Path | Mapping[str, Any]) -> None:
        if isinstance(source, Mapping):
            payload = dict(source)
        else:
            payload = self._load_file(Path(source))
        datasets = payload.get("datasets", payload)
        if not isinstance(datasets, Mapping):
            raise ValueError("Catalog must contain a 'datasets' mapping")
        self._datasets = {str(name): dict(spec) for name, spec in datasets.items()}

    def resolve(self, name: str) -> DatasetSpec:
        if name not in self._datasets:
            raise KeyError(f"Dataset not found in catalog: {name}")
        payload = self._datasets[name]
        return _normalize_spec(name, payload)

    def get_source(self, name: str) -> SourceConfig:
        return self.resolve(name).to_source()

    def _load_file(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix == ".json":
            return json.loads(path.read_text())
        if path.suffix == ".toml":
            return tomllib.loads(path.read_text())
        raise ValueError(f"Unsupported catalog file type: {path.suffix}")


def _normalize_spec(name: str, payload: Mapping[str, Any]) -> DatasetSpec:
    fmt = payload.get("format") or payload.get("type")
    if fmt is None:
        raise ValueError(f"Dataset '{name}' is missing 'format'")
    path = payload.get("path") or payload.get("location")
    if path is None:
        raise ValueError(f"Dataset '{name}' is missing 'path'")

    reserved = {
        "format",
        "type",
        "path",
        "location",
        "options",
    }
    extra_options = {key: value for key, value in payload.items() if key not in reserved}
    options = dict(payload.get("options") or {})
    options.update(extra_options)

    return DatasetSpec(
        name=str(name),
        format=str(fmt),
        path=path,
        options=options,
    )
