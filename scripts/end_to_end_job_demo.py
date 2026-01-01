from __future__ import annotations

import os
import time
from pathlib import Path
from uuid import uuid4

import polars as pl
import polars_incremental as pli


def write_raw_batch(raw_dir: Path, batch_id: int, rows: int = 3) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "event_id": [str(uuid4()) for _ in range(rows)],
        "batch_id": [batch_id] * rows,
        "seq": list(range(rows)),
        "value": [float(batch_id * 10 + i) for i in range(rows)],
        "event_ts": [time.time()] * rows,
    }
    df = pl.DataFrame(data)
    path = raw_dir / f"events_{batch_id}_{uuid4().hex}.parquet"
    df.write_parquet(path)
    print(f"wrote raw file: {path}")


def main() -> None:
    base_dir = Path("data/e2e_job_demo")
    raw_dir = base_dir / "raw"
    delta_dir = base_dir / "delta"
    checkpoint_dir = base_dir / "checkpoints"

    def reader(files: list[str]):
        return pl.scan_parquet(files)

    def transform(lf: pl.LazyFrame):
        return lf.with_columns((pl.col("value") * 2).alias("value_x2"))

    def writer(lf: pl.LazyFrame, batch_id: int | None = None):
        df = lf.collect()
        df.write_delta(str(delta_dir), mode="append")
        print(f"wrote batch {batch_id} rows={df.height}")

    write_raw_batch(raw_dir, batch_id=0)
    write_raw_batch(raw_dir, batch_id=0, rows=2)

    pipeline = pli.Pipeline(
        source=pli.FilesSource(
            path=raw_dir,
            file_format="parquet",
            pattern="events_*.parquet",
        ),
        checkpoint_dir=checkpoint_dir,
        reader=reader,
        transform=transform,
        writer=writer,
    )
    pipeline.run(once=True)

    write_raw_batch(raw_dir, batch_id=1)
    write_raw_batch(raw_dir, batch_id=1, rows=4)

    pipeline.run(once=True)

    try:
        compact_result = pli.optimize_delta_table(delta_dir, mode="compact")
        print("compact result:", compact_result)
        z_order_result = pli.optimize_delta_table(
            delta_dir, mode="z_order", z_order_columns=["batch_id"]
        )
        print("z_order result:", z_order_result)
    except RuntimeError as exc:
        print("optimize not available:", exc)

    # Overwrite the table to create tombstones for older files.
    overwrite_df = pl.DataFrame(
        {
            "event_id": ["reset"],
            "batch_id": [-1],
            "seq": [0],
            "value": [0.0],
            "event_ts": [time.time()],
            "value_x2": [0.0],
        }
    )
    parquet_before = list(delta_dir.rglob("*.parquet"))
    overwrite_df.write_delta(str(delta_dir), mode="overwrite")
    parquet_after_overwrite = list(delta_dir.rglob("*.parquet"))
    print(
        "parquet files (before overwrite -> after overwrite):",
        len(parquet_before),
        "->",
        len(parquet_after_overwrite),
    )

    # Backdate parquet files to make vacuum behavior visible in the demo.
    backdate_seconds = 8 * 24 * 60 * 60
    backdate_ts = time.time() - backdate_seconds
    for path in delta_dir.rglob("*.parquet"):
        os.utime(path, (backdate_ts, backdate_ts))

    # Non-dry vacuum; disable retention enforcement and use 0 hours to force cleanup.
    vacuum_result = pli.vacuum_delta_table(
        delta_dir,
        retention_hours=0,
        dry_run=False,
        enforce_retention=False,
    )
    print("vacuum result:", vacuum_result)
    parquet_after_vacuum = list(delta_dir.rglob("*.parquet"))
    print("parquet files after vacuum:", len(parquet_after_vacuum))

    cleanup = pli.cleanup_checkpoint(checkpoint_dir, keep_last_n=2)
    print("checkpoint cleanup:", cleanup)


if __name__ == "__main__":
    main()
