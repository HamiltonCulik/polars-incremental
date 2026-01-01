"""Example: Observability with LoggingObserver."""

from pathlib import Path
import logging
import shutil

import polars as pl
import polars_incremental as pli

base_dir = Path("data/observability_basic")
raw_dir = base_dir / "raw"
checkpoint_dir = base_dir / "checkpoint"

if base_dir.exists():
    shutil.rmtree(base_dir)

raw_dir.mkdir(parents=True, exist_ok=True)

pl.DataFrame({"id": [1, 2], "value": [10, 20]}).write_parquet(raw_dir / "part-0000.parquet")

def reader(files):
    return pl.scan_parquet(files)

def writer(lf, batch_id=None):
    df = lf.collect()
    df.write_parquet(base_dir / f"out_{batch_id}.parquet")
    return {"rows_out": df.height}

logging.basicConfig(level=logging.INFO)

pipeline = pli.Pipeline(
    source=pli.FilesSource(path=raw_dir, file_format="parquet"),
    checkpoint_dir=checkpoint_dir,
    reader=reader,
    writer=writer,
    observer=pli.LoggingObserver(),
)

pipeline.run(once=True)
