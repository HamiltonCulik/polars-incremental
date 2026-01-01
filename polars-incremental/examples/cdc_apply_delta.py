"""Example: apply CDC rows to a Delta sink.

Scenario: a CDC feed (with _change_type) is read from files and merged into a Delta table.
"""
from pathlib import Path

import polars as pl
import polars_incremental as pli

base = Path("data/cdc_apply_example")
raw_dir = base / "cdc_raw"
checkpoint_dir = base / "checkpoint"
target_dir = base / "delta"

raw_dir.mkdir(parents=True, exist_ok=True)

# Seed the target table
pl.DataFrame({"id": [1, 2], "value": [10, 20]}).write_delta(target_dir, mode="overwrite")

# CDC batch (insert, update, delete)
pl.DataFrame(
    {
        "id": [2, 1, 3],
        "value": [25, None, 30],
        "_change_type": ["update_postimage", "delete", "insert"],
        "_commit_version": [1, 1, 1],
    }
).write_parquet(raw_dir / "batch-0000.parquet")


def reader(files):
    return pl.read_parquet(files)


def writer(df, batch_id=None):
    return pli.apply_cdc_delta(df, target_dir, keys=["id"])


pipeline = pli.Pipeline(
    source=pli.FilesSource(path=raw_dir, file_format="parquet"),
    checkpoint_dir=checkpoint_dir,
    reader=reader,
    writer=writer,
)

pipeline.run(once=True)

print(pl.read_delta(str(target_dir)).sort("id"))
