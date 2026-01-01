"""Example: File source with start offsets.

Scenario: You have a landing folder and want to control where ingestion starts.
"""

from pathlib import Path

import polars as pl
import polars_incremental as pli

base_dir = Path("data/files_start_offset_example")
raw_dir = base_dir / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

pl.DataFrame({"id": [1], "value": [10]}).write_parquet(raw_dir / "part-0000.parquet")
pl.DataFrame({"id": [2], "value": [20]}).write_parquet(raw_dir / "part-0001.parquet")

def reader(files):
    return pl.scan_parquet(files)

def writer_factory(out_dir: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def writer(lf, batch_id=None):
        lf.sink_parquet(out_path / f"batch_{batch_id}.parquet")
    return writer

# Example 1: start at latest (skip existing, only new arrivals)
pipeline = pli.Pipeline(
    source=pli.FilesSource(path=raw_dir, start_offset="latest"),
    checkpoint_dir=base_dir / "checkpoints" / "latest",
    reader=reader,
    writer=writer_factory(str(base_dir / "out" / "parquet_sink_latest")),
)

pipeline.run(once=True)

# Example 2: start at earliest (process everything)
pipeline = pli.Pipeline(
    source=pli.FilesSource(path=raw_dir, start_offset="earliest"),
    checkpoint_dir=base_dir / "checkpoints" / "earliest",
    reader=reader,
    writer=writer_factory(str(base_dir / "out" / "parquet_sink_earliest")),
)

pipeline.run(once=True)

# Example 3: start from a timestamp (ISO string or unix seconds)
pipeline = pli.Pipeline(
    source=pli.FilesSource(path=raw_dir, start_timestamp="2025-12-31T00:00:00Z"),
    checkpoint_dir=base_dir / "checkpoints" / "timestamp",
    reader=reader,
    writer=writer_factory(str(base_dir / "out" / "parquet_sink_timestamp")),
)

pipeline.run(once=True)
