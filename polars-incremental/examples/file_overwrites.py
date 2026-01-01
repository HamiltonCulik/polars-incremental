"""Example: Allow overwrites in a landing folder.

Scenario: Files can be re-written with the same name; enable allow_overwrites.
"""

from pathlib import Path

import polars as pl
import polars_incremental as pli

base_dir = Path("data/file_overwrites_example")
raw_dir = base_dir / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

pl.DataFrame({"id": [1, 2], "value": [10, 20]}).write_parquet(
    raw_dir / "part-0000.parquet"
)

def reader(files):
    return pl.scan_parquet(files)

def writer_factory(out_dir: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def writer(lf, batch_id=None):
        lf.sink_parquet(out_path / f"batch_{batch_id}.parquet")
    return writer

# Example 1: allow overwrites (reprocess file if mtime/size changes)
pipeline = pli.Pipeline(
    source=pli.FilesSource(path=raw_dir, allow_overwrites=True),
    checkpoint_dir=base_dir / "checkpoints" / "overwrites",
    reader=reader,
    writer=writer_factory(str(base_dir / "out" / "parquet_sink_overwrites")),
)

pipeline.run(once=True)

# Example 2: default behavior (skip previously processed files)
pipeline = pli.Pipeline(
    source=pli.FilesSource(path=raw_dir),
    checkpoint_dir=base_dir / "checkpoints" / "default",
    reader=reader,
    writer=writer_factory(str(base_dir / "out" / "parquet_sink_default")),
)

pipeline.run(once=True)
