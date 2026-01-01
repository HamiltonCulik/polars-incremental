"""Example: Delta source -> Delta sink with optional CDF.

Scenario: You have a Delta table of events and want to stream changes into a sink table.
"""

from pathlib import Path
import shutil

import polars as pl
import polars_incremental as pli

base_dir = Path("data/delta_cdf_example")
if base_dir.exists():
    shutil.rmtree(base_dir)
source_dir = base_dir / "source_events"
source_dir.mkdir(parents=True, exist_ok=True)

# Seed a Delta table with CDF enabled, then append changes.
pl.DataFrame({"id": [1, 2], "value": [10, 20]}).write_delta(
    source_dir,
    mode="append",
    delta_write_options={"configuration": {"delta.enableChangeDataFeed": "true"}},
)
pl.DataFrame({"id": [3], "value": [30]}).write_delta(source_dir, mode="append")

def reader(files, batch=None):
    if batch is not None and getattr(batch, "file_entries", None) is not None:
        return pli.read_cdf_batch(batch)
    return pl.read_parquet(files)

def writer(df, batch=None):
    df.write_delta(base_dir / "sink_events_cdf", mode="append")

# Example 1: read from CDF starting at version 0
pipeline = pli.Pipeline(
    source=pli.DeltaSource(
        path=source_dir,
        starting_version=0,
        read_change_feed=True,
    ),
    checkpoint_dir=base_dir / "checkpoints" / "delta_source_cdf",
    reader=reader,
    writer=writer,
)

pipeline.run(once=True)

# Example 2: read latest snapshot (no CDF)
def writer_latest(df, batch=None):
    df.write_delta(base_dir / "sink_events_latest", mode="append")

pipeline = pli.Pipeline(
    source=pli.DeltaSource(
        path=source_dir,
        start_offset="latest",
    ),
    checkpoint_dir=base_dir / "checkpoints" / "delta_source_latest",
    reader=reader,
    writer=writer_latest,
)

pipeline.run(once=True)
