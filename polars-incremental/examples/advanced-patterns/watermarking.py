from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import polars_incremental as pli

base_dir = Path("data/watermarking_demo")
raw_dir = base_dir / "raw"
out_dir = base_dir / "out"
checkpoint_dir = base_dir / "checkpoint"
allowed_lateness = timedelta(minutes=5)

if base_dir.exists():
    shutil.rmtree(base_dir)

raw_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)

pl.DataFrame(
    {
        "id": [1, 2, 3],
        "value": [10, 20, 30],
        "event_time": ["2024-01-01T10:00:00", "2024-01-01T10:03:00", "2024-01-01T10:05:00"],
    }
).write_parquet(raw_dir / "batch-0000.parquet")

pl.DataFrame(
    {
        "id": [4, 5, 6],
        "value": [40, 50, 60],
        "event_time": ["2024-01-01T10:02:00", "2024-01-01T09:40:00", "2024-01-01T10:08:00"],
    }
).write_parquet(raw_dir / "batch-0001.parquet")


def reader(files):
    return pl.read_parquet(files)


def transform(df, state=None):
    df = df.with_columns(pl.col("event_time").str.strptime(pl.Datetime, strict=False))
    if state is not None:
        watermark_payload = state.load_json("watermark", default={"value": None})
        watermark_value = watermark_payload.get("value")
    else:
        watermark_value = None
    watermark = datetime.fromisoformat(watermark_value) if watermark_value else None
    if watermark is None:
        accepted = df
        late = df.head(0)
    else:
        cutoff = watermark - allowed_lateness
        accepted = df.filter(pl.col("event_time") >= pl.lit(cutoff))
        late = df.filter(pl.col("event_time") < pl.lit(cutoff))
    max_event_time = df.select(pl.max("event_time")).item()
    return {"accepted": accepted, "late": late, "max_event_time": max_event_time}


def writer(payload, batch_id=None, state=None):
    accepted = payload["accepted"]
    late = payload["late"]
    max_event_time = payload["max_event_time"]

    out_path = out_dir / f"accepted_{batch_id}.parquet"
    accepted.write_parquet(out_path)

    if late.height:
        print(f"batch {batch_id}: dropped {late.height} late rows")
    else:
        print(f"batch {batch_id}: no late rows")

    if max_event_time is not None:
        if state is not None:
            current_payload = state.load_json("watermark", default={"value": None})
            current_value = current_payload.get("value")
        else:
            current_value = None
        current = datetime.fromisoformat(current_value) if current_value else None
        next_watermark = max_event_time if current is None else max(current, max_event_time)
        if state is not None:
            state.save_json("watermark", {"value": next_watermark.isoformat()})

    return {"rows_out": accepted.height, "late_dropped": late.height}


pipeline = pli.Pipeline(
    source=pli.FilesSource(path=raw_dir, file_format="parquet", max_files_per_trigger=1),
    checkpoint_dir=checkpoint_dir,
    reader=reader,
    transform=transform,
    writer=writer,
    observer=pli.LoggingObserver(),
)

pipeline.run(once=False, max_batches=2, sleep=0.0)

state = pli.JobState(checkpoint_dir / "state")
wm_payload = state.load_json("watermark", default={"value": None})
print("watermark:", wm_payload.get("value"))

# Example cleanup after a migration/rename.
if state.exists("watermark_old", kind="json"):
    state.delete("watermark_old", kind="json")
accepted_rows = 0
for path in sorted(out_dir.glob("accepted_*.parquet")):
    accepted_rows += pl.read_parquet(path).height
print("accepted rows:", accepted_rows)
