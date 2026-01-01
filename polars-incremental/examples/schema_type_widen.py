"""Example: type_widen schema evolution.

Scenario: stored schema expects Int64; incoming data widens to Float64/String.
"""
from pathlib import Path
import shutil

import polars as pl
import polars_incremental as pli

base = Path("data/schema_type_widen")
raw_dir = base / "raw"
checkpoint_dir = base / "checkpoint"
out_dir = base / "out"

if base.exists():
    shutil.rmtree(base)

raw_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)

# Batch 0: int schema
pl.DataFrame({"a": [1, 2], "b": [10, 20]}).write_parquet(raw_dir / "batch-0000.parquet")

# Batch 1: widening to float + string
pl.DataFrame({"a": [1.5, 2.25], "b": ["10", "oops"]}).write_parquet(
    raw_dir / "batch-0001.parquet"
)


def reader(files):
    frames = [pl.read_parquet(path) for path in files]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def writer(df, batch_id=None):
    df.write_parquet(out_dir / f"batch_{batch_id}.parquet")


pipeline = pli.Pipeline(
    source=pli.FilesSource(
        path=raw_dir,
        file_format="parquet",
        max_files_per_trigger=1,
    ),
    checkpoint_dir=checkpoint_dir,
    reader=reader,
    writer=writer,
    schema_evolution=pli.SchemaEvolution(mode="type_widen"),
)

# First run seeds the schema
pipeline.run(once=True)

# Second run widens types as needed
pipeline.run(once=True)
