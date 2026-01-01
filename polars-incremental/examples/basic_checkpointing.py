"""Example: Basic checkpointed ingestion (minimal usage).

Scenario: Run the same job repeatedly; only new files are processed.
"""

from pathlib import Path

import polars as pl
import polars_incremental as pli

base_dir = Path("data/basic_checkpointing")
raw_dir = base_dir / "raw"
delta_dir = base_dir / "delta" / "basic_events"
raw_dir.mkdir(parents=True, exist_ok=True)

# Seed a small raw batch for the examples.
pl.DataFrame({"id": [1, 2], "value": [10, 20], "amount": [10, 20]}).write_parquet(
    raw_dir / "part-0000.parquet"
)

# Example 1: minimal files -> parquet sink
def reader(files):
    return pl.scan_parquet(files)

def parquet_writer_factory(out_dir: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def writer(lf, batch_id=None):
        lf.sink_parquet(out_path / f"batch_{batch_id}.parquet")
    return writer

pipeline = pli.Pipeline(
    source=pli.FilesSource(path=raw_dir, file_format="parquet"),
    checkpoint_dir=base_dir / "checkpoints" / "basic_parquet",
    reader=reader,
    writer=parquet_writer_factory(str(base_dir / "out" / "basic_parquet")),
)

pipeline.run(once=True)

# Example 2: files -> delta sink with a transform
def transform_amount(lf):
    return lf.with_columns(pl.col("amount").cast(pl.Float64))

def delta_writer(lf, batch=None):
    df = lf.collect()
    df.write_delta(delta_dir, mode="append")

pipeline = pli.Pipeline(
    source=pli.FilesSource(path=raw_dir, file_format="parquet"),
    checkpoint_dir=base_dir / "checkpoints" / "basic_delta",
    reader=reader,
    transform=transform_amount,
    writer=delta_writer,
)

pipeline.run(once=True)

# Example 3: start at latest (skip existing files)
pipeline = pli.Pipeline(
    source=pli.FilesSource(path=raw_dir, start_offset="latest"),
    checkpoint_dir=base_dir / "checkpoints" / "basic_latest",
    reader=reader,
    writer=parquet_writer_factory(str(base_dir / "out" / "basic_latest")),
)

pipeline.run(once=True)
