from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from ..checkpoints.delta import DeltaTableCheckpoint
from ..checkpoints.types import DeltaBatch
from ..utils.options import get_option
from .base import Source


def read_cdf_batch(batch: DeltaBatch) -> pl.DataFrame:
    if not batch.files:
        return pl.DataFrame()
    if batch.file_entries is None:
        return pl.read_parquet(batch.files)

    frames = []
    for entry in batch.file_entries:
        df = pl.read_parquet(entry.path)
        if entry.change_type and "_change_type" not in df.columns:
            df = df.with_columns(pl.lit(entry.change_type).alias("_change_type"))
        if "_commit_version" not in df.columns:
            df = df.with_columns(pl.lit(entry.commit_version).alias("_commit_version"))
        if entry.commit_timestamp is not None and "_commit_timestamp" not in df.columns:
            df = df.with_columns(pl.lit(entry.commit_timestamp).alias("_commit_timestamp"))
        frames.append(df)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical")


class DeltaSource(Source):
    def __init__(self, path: str | Path, checkpoint_dir: str | Path, options: dict[str, Any]) -> None:
        self.path = Path(path)
        self.checkpoint = DeltaTableCheckpoint(checkpoint_dir)
        self.start_offset = get_option(options, "start_offset", default=None)
        self.starting_version = get_option(options, "starting_version", default=None)
        self.starting_timestamp = get_option(options, "starting_timestamp", default=None)
        self.max_files = get_option(options, "max_files_per_trigger", default=1000)
        self.max_bytes = get_option(options, "max_bytes_per_trigger", default=None)
        self.ignore_deletes = bool(get_option(options, "ignore_deletes", default=False))
        self.ignore_changes = bool(get_option(options, "ignore_changes", default=False))
        self.read_change_feed = bool(get_option(options, "read_change_feed", default=False))
    def plan_batch(self) -> DeltaBatch | None:
        batch = self.checkpoint.plan_batch(
            self.path,
            start_offset=self.start_offset,
            starting_version=self.starting_version,
            starting_timestamp=self.starting_timestamp,
            max_files=self.max_files,
            max_bytes=self.max_bytes,
            ignore_deletes=self.ignore_deletes,
            ignore_changes=self.ignore_changes,
            read_change_feed=self.read_change_feed,
        )
        if batch is None:
            return None
        self.checkpoint.write_offset(batch)
        return batch

    def read_batch(self, batch: DeltaBatch) -> pl.DataFrame:
        if not batch.files:
            return pl.DataFrame()
        if self.read_change_feed:
            return read_cdf_batch(batch)
        else:
            return pl.read_parquet(batch.files)

    def commit_batch(self, batch: DeltaBatch, metadata: dict | None = None) -> None:
        self.checkpoint.commit_batch(batch, metadata=metadata)
