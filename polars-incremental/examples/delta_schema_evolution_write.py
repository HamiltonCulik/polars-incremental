"""Example: Delta sink schema evolution (mergeSchema).

Scenario: A new column appears and a numeric column widens from int -> float.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import polars_incremental as pli

base = Path("data/delta_schema_evolution_example")
raw_dir = base / "raw"
checkpoint_dir = base / "checkpoint"
delta_dir = base / "delta"

raw_dir.mkdir(parents=True, exist_ok=True)

# Batch 0: initial schema (amount is integer)
pl.DataFrame({"id": [1, 2], "amount": [10, 20]}).write_parquet(
    raw_dir / "batch-0000.parquet"
)

# Coerce amount to Float64 so type drift is handled before the Delta append.
def reader(files):
    return pl.read_parquet(files)

def writer(df, batch=None):
    df.write_delta(
        delta_dir / "merge",
        mode="append",
        delta_write_options={"schema_mode": "merge"},
    )

schema_evolution = pli.SchemaEvolution(
    mode="coerce",
    schema={"amount": "Float64"},
)
pipeline = pli.Pipeline(
    source=pli.FilesSource(path=raw_dir, file_format="parquet"),
    checkpoint_dir=checkpoint_dir / "merge",
    reader=reader,
    writer=writer,
    schema_evolution=schema_evolution,
)

pipeline.run(once=True)

# Batch 1: schema drift (new column + float amount)
pl.DataFrame({"id": [3], "amount": [10.5], "note": ["late"]}).write_parquet(
    raw_dir / "batch-0001.parquet"
)

pipeline.run(once=True)

# Example 2: strict schema with consistent input (no evolution expected)
strict_raw = base / "raw_strict"
strict_raw.mkdir(parents=True, exist_ok=True)
pl.DataFrame({"id": [10, 11], "amount": [30, 40]}).write_parquet(
    strict_raw / "batch-0000.parquet"
)
pl.DataFrame({"id": [12], "amount": [50]}).write_parquet(
    strict_raw / "batch-0001.parquet"
)

def writer_strict(df, batch=None):
    df.write_delta(delta_dir / "strict", mode="append")

strict_pipeline = pli.Pipeline(
    source=pli.FilesSource(path=strict_raw, file_format="parquet"),
    checkpoint_dir=checkpoint_dir / "strict",
    reader=reader,
    writer=writer_strict,
    schema_evolution=pli.SchemaEvolution(mode="strict"),
)

strict_pipeline.run(once=True)

table = pl.read_delta(str(delta_dir / "merge"))
print(table)
print(table.schema)
