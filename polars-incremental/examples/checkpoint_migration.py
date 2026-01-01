"""Example: Checkpoint migration (truncate and reprocess)."""

from pathlib import Path
import shutil

import polars as pl
import polars_incremental as pli

base_dir = Path("data/checkpoint_migration")
raw_dir = base_dir / "raw"
out_dir = base_dir / "out"
checkpoint_dir = base_dir / "checkpoint"

if base_dir.exists():
    shutil.rmtree(base_dir)

raw_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)

pl.DataFrame({"id": [1], "value": [10]}).write_parquet(raw_dir / "part-0000.parquet")
pl.DataFrame({"id": [2], "value": [20]}).write_parquet(raw_dir / "part-0001.parquet")

write_count = {"n": 0}


def reader(files):
    return pl.read_parquet(files)


def writer(df, batch_id=None):
    write_count["n"] += 1
    path = out_dir / f"batch_{batch_id}_run_{write_count['n']}.parquet"
    df.write_parquet(path)
    print(f"wrote {path.name}")


pipeline = pli.Pipeline(
    source=pli.FilesSource(path=raw_dir, file_format="parquet", max_files_per_trigger=1),
    checkpoint_dir=checkpoint_dir,
    reader=reader,
    writer=writer,
)

# First pass: two batches (one file per batch).
pipeline.run(once=True)
pipeline.run(once=True)

print("checkpoint before truncate:", pli.inspect_checkpoint(checkpoint_dir))

# Truncate after batch 0, so batch 1 is eligible to reprocess.
pli.truncate_checkpoint(checkpoint_dir, keep_through_batch_id=0)

# Reprocess the second file (batch_id=1 again).
pipeline.run(once=True)

print("checkpoint after truncate:", pli.inspect_checkpoint(checkpoint_dir))
print("output files:", sorted(path.name for path in out_dir.glob("*.parquet")))
