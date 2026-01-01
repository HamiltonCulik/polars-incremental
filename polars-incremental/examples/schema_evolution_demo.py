"""Example: generate mock data and run schema evolution with rescue.

Scenario: Mixed types and new columns appear in raw files.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import polars_incremental as pli

base = Path("data/schema_evolution_demo")
raw_dir = base / "raw"
checkpoint_dir = base / "checkpoint"
sink_dir = base / "sink"

raw_dir.mkdir(parents=True, exist_ok=True)

# Write two raw files with schema drift
pl.DataFrame({"a": [1, 2, 3]}).write_parquet(raw_dir / "part-0000.parquet")
pl.DataFrame({"a": ["4", "bad", "6"], "b": ["x", "y", "z"]}).write_parquet(
    raw_dir / "part-0001.parquet"
)

def reader(files):
    frames = [pl.read_parquet(path) for path in files]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

def writer_factory(out_dir: Path):
    def writer(df, batch_id=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_dir / f"batch_{batch_id}.parquet")
    return writer

# Example 1: coerce 'a' to Int64 and rescue failed casts into _rescued
pipeline = pli.Pipeline(
    source=pli.FilesSource(path=raw_dir),
    checkpoint_dir=checkpoint_dir / "coerce",
    reader=reader,
    writer=writer_factory(sink_dir / "coerce"),
    schema_evolution=pli.SchemaEvolution(
        mode="coerce",
        schema={"a": "Int64"},
        rescue_mode="column",
        rescue_column="_rescued",
    ),
)

pipeline.run(once=True)

# Example 2: add new columns without coercion
pipeline = pli.Pipeline(
    source=pli.FilesSource(path=raw_dir),
    checkpoint_dir=checkpoint_dir / "add_cols",
    reader=reader,
    writer=writer_factory(sink_dir / "add_cols"),
    schema_evolution=pli.SchemaEvolution(mode="add_new_columns"),
)

pipeline.run(once=True)

# Inspect the written output
result = pl.read_parquet(sink_dir / "coerce" / "batch_0.parquet")
print(result)
print(result.select(pl.col("_rescued").struct.field("a")).to_series().to_list())
