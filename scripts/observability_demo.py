from __future__ import annotations

import logging
import shutil
from pathlib import Path

import polars as pl
import polars_incremental as pli

base_dir = Path("data/observability_demo")
raw_dir = base_dir / "raw"
out_dir = base_dir / "out"
checkpoint_dir = base_dir / "checkpoint"

if base_dir.exists():
    shutil.rmtree(base_dir)

raw_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)

pl.DataFrame({"id": [1, 2], "value": [10, 20]}).write_parquet(raw_dir / "part-0000.parquet")
pl.DataFrame({"id": [3], "value": [30]}).write_parquet(raw_dir / "part-0001.parquet")


def reader(files):
    return pl.scan_parquet(files)


def writer(lf, batch_id=None):
    df = lf.collect()
    out_path = out_dir / f"batch_{batch_id}.parquet"
    df.write_parquet(out_path)
    print(f"wrote {out_path.name} rows={df.height}")
    return {"rows_out": df.height, "out_path": str(out_path)}


logging.basicConfig(level=logging.INFO)

pipeline = pli.Pipeline(
    source=pli.FilesSource(path=raw_dir, file_format="parquet", max_files_per_trigger=1),
    checkpoint_dir=checkpoint_dir,
    reader=reader,
    writer=writer,
    observer=pli.LoggingObserver(),
)

pipeline.run(once=False, max_batches=2, sleep=0.0)

print("checkpoint:", pli.inspect_checkpoint(checkpoint_dir))
print("output files:", sorted(path.name for path in out_dir.glob("*.parquet")))
print(pl.read_parquet(out_dir / "batch_0.parquet"))
