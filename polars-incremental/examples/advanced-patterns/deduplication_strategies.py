from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import polars_incremental as pli

base_dir = Path("data/deduplication_strategies_demo")
raw_dir = base_dir / "raw"
out_dir_a = base_dir / "out_id_dedup"
out_dir_b = base_dir / "out_latest_per_user"
checkpoint_a = base_dir / "checkpoint_id_dedup"
checkpoint_b = base_dir / "checkpoint_latest_per_user"

if base_dir.exists():
    shutil.rmtree(base_dir)

raw_dir.mkdir(parents=True, exist_ok=True)
out_dir_a.mkdir(parents=True, exist_ok=True)
out_dir_b.mkdir(parents=True, exist_ok=True)

pl.DataFrame(
    {
        "event_id": [1, 1, 2, 3],
        "user_id": [101, 101, 102, 103],
        "value": [10, 11, 20, 30],
        "event_time": [
            "2024-01-01T00:00:00",
            "2024-01-01T00:00:01",
            "2024-01-01T00:02:00",
            "2024-01-01T00:03:00",
        ],
    }
).write_parquet(raw_dir / "batch-0000.parquet")

pl.DataFrame(
    {
        "event_id": [2, 4, 5, 5],
        "user_id": [102, 104, 105, 105],
        "value": [21, 40, 50, 51],
        "event_time": [
            "2024-01-01T00:02:30",
            "2024-01-01T00:04:00",
            "2024-01-01T00:05:00",
            "2024-01-01T00:05:01",
        ],
    }
).write_parquet(raw_dir / "batch-0001.parquet")


def reader(files):
    return pl.read_parquet(files)


# Strategy A: id-based dedupe (within-batch + cross-batch).
# Cross-batch state is persisted via JobState.


def transform_id_dedupe(df, state=None):
    df = df.with_columns(pl.col("event_time").str.strptime(pl.Datetime, strict=False))
    df = df.sort("event_time").unique(subset=["event_id"], keep="last")
    seen_list = state.load_json("seen_ids", default=[]) if state is not None else []
    seen = {int(x) for x in seen_list}
    new = df.filter(~pl.col("event_id").is_in(list(seen)))
    return {"deduped": new, "seen_before": len(seen)}


def writer_id_dedupe(payload, batch_id=None, state=None):
    deduped = payload["deduped"]
    seen_list = state.load_json("seen_ids", default=[]) if state is not None else []
    seen = {int(x) for x in seen_list}
    new_ids = set(deduped["event_id"].to_list()) if deduped.height else set()
    if state is not None:
        state.save_json("seen_ids", sorted(seen | new_ids))
    out_path = out_dir_a / f"batch_{batch_id}.parquet"
    deduped.write_parquet(out_path)
    print(f"id-dedupe batch {batch_id}: wrote {deduped.height} rows")
    return {"rows_out": deduped.height, "new_ids": len(new_ids)}


pipeline_a = pli.Pipeline(
    source=pli.FilesSource(path=raw_dir, file_format="parquet", max_files_per_trigger=1),
    checkpoint_dir=checkpoint_a,
    reader=reader,
    transform=transform_id_dedupe,
    writer=writer_id_dedupe,
    observer=pli.LoggingObserver(),
)

pipeline_a.run(once=False, max_batches=2, sleep=0.0)


# Strategy B: keep latest per user (upsert by event_time).
# The "latest table" is maintained in JobState and also written to a public output.
latest_path = out_dir_b / "latest_per_user.parquet"


def transform_latest_per_user(df):
    return df.with_columns(pl.col("event_time").str.strptime(pl.Datetime, strict=False))


def writer_latest_per_user(df, batch_id=None, state=None):
    batch_latest = (
        df.sort("event_time")
        .unique(subset=["user_id"], keep="last")
        .select(["user_id", "event_time", "event_id", "value"])
    )
    existing = state.load_parquet("latest_per_user") if state is not None else None
    if existing is None and latest_path.exists():
        existing = pl.read_parquet(latest_path)
    if existing is None:
        final = batch_latest.sort("user_id")
    else:
        combined = pl.concat([existing, batch_latest], how="diagonal")
        final = (
            combined.sort("event_time")
            .unique(subset=["user_id"], keep="last")
            .sort("user_id")
        )
    if state is not None:
        state.save_parquet("latest_per_user", final)
    final.write_parquet(latest_path)
    print(f"latest-per-user batch {batch_id}: users={final.height}")
    return {"rows_out": batch_latest.height}


pipeline_b = pli.Pipeline(
    source=pli.FilesSource(path=raw_dir, file_format="parquet", max_files_per_trigger=1),
    checkpoint_dir=checkpoint_b,
    reader=reader,
    transform=transform_latest_per_user,
    writer=writer_latest_per_user,
    observer=pli.LoggingObserver(),
)

pipeline_b.run(once=False, max_batches=2, sleep=0.0)

print("final id-dedupe rows:", sum(pl.read_parquet(path).height for path in out_dir_a.glob("batch_*.parquet")))
print("final latest-per-user table:")
print(pl.read_parquet(latest_path))
