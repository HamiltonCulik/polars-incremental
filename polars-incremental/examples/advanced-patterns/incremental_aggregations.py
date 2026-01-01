from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import polars_incremental as pli

base_dir = Path("data/incremental_aggregations_demo")
raw_dir = base_dir / "raw"
out_dir = base_dir / "out"
checkpoint_dir = base_dir / "checkpoint"

if base_dir.exists():
    shutil.rmtree(base_dir)

raw_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)

pl.DataFrame(
    {
        "user_id": [101, 102, 101],
        "amount": [10.0, 7.5, 2.5],
        "event_time": ["2024-01-01T00:00:00", "2024-01-01T00:02:00", "2024-01-01T00:03:00"],
    }
).write_parquet(raw_dir / "batch-0000.parquet")

pl.DataFrame(
    {
        "user_id": [102, 103, 101],
        "amount": [5.0, 12.0, 1.0],
        "event_time": ["2024-01-01T00:06:00", "2024-01-01T00:07:00", "2024-01-01T00:08:00"],
    }
).write_parquet(raw_dir / "batch-0001.parquet")


def reader(files):
    return pl.read_parquet(files)


def transform(df):
    return df.with_columns(
        pl.col("event_time").str.strptime(pl.Datetime, strict=False)
    )


def writer(df, batch_id=None, state=None):
    # Rolling aggregates are kept in JobState and written to a public output table.
    batch_agg = (
        df.group_by("user_id")
        .agg(
            pl.sum("amount").alias("total_amount"),
            pl.len().alias("txn_count"),
        )
        .sort("user_id")
    )
    existing = state.load_parquet("aggregates") if state is not None and state.exists("aggregates") else None
    agg_path = out_dir / "aggregates.parquet"
    if existing is None and agg_path.exists():
        existing = pl.read_parquet(agg_path)
    if existing is not None:
        combined = pl.concat([existing, batch_agg], how="diagonal")
        final = (
            combined.group_by("user_id")
            .agg(
                pl.sum("total_amount").alias("total_amount"),
                pl.sum("txn_count").alias("txn_count"),
            )
            .sort("user_id")
        )
    else:
        final = batch_agg
    if state is not None:
        state.save_parquet("aggregates", final)
    final.write_parquet(agg_path)
    print(f"batch {batch_id}: updated {batch_agg.height} users")
    return {"rows_in": df.height, "users_updated": batch_agg.height}


pipeline = pli.Pipeline(
    source=pli.FilesSource(path=raw_dir, file_format="parquet", max_files_per_trigger=1),
    checkpoint_dir=checkpoint_dir,
    reader=reader,
    transform=transform,
    writer=writer,
    observer=pli.LoggingObserver(),
)

# Process each file as a separate micro-batch.
pipeline.run(once=False, max_batches=2, sleep=0.0)

print("final aggregates:")
print(pl.read_parquet(out_dir / "aggregates.parquet"))
