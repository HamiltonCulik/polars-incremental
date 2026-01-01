from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

import polars as pl

import polars_incremental as pli


def write_raw_batch(raw_dir: Path, batch_id: int, rows: int = 4) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "event_id": [str(uuid4()) for _ in range(rows)],
        "batch_id": [batch_id] * rows,
        "user_id": [f"user_{i % 3}" for i in range(rows)],
        "amount": [float(batch_id * 10 + i) for i in range(rows)],
        "event_ts": [time.time()] * rows,
    }
    df = pl.DataFrame(data)
    path = raw_dir / f"events_{batch_id}_{uuid4().hex}.parquet"
    df.write_parquet(path)
    print(f"wrote raw file: {path}")


def main() -> None:
    base_dir = Path("data/sql_end_to_end_demo")
    raw_dir = base_dir / "raw"
    out_dir = base_dir / "out"
    checkpoint_dir = base_dir / "checkpoints"

    out_dir.mkdir(parents=True, exist_ok=True)

    def reader(files: list[str]) -> pl.LazyFrame:
        return pl.scan_parquet(files)

    def transform(lf: pl.LazyFrame) -> pl.LazyFrame:
        ctx = pl.SQLContext()
        ctx.register("events", lf)
        return ctx.execute(
            """
            SELECT
                batch_id,
                user_id,
                COUNT(*) AS event_count,
                SUM(amount) AS total_amount
            FROM events
            WHERE amount >= 0
            GROUP BY batch_id, user_id
            ORDER BY batch_id, user_id
            """,
            eager=False,
        )

    def writer(
        lf: pl.LazyFrame,
        batch_id: int | None = None,
        state: pli.JobState | None = None,
    ) -> None:
        df = lf.collect()
        df.write_parquet(out_dir / f"batch_{batch_id}.parquet")

        previous = None if state is None else state.load_parquet("user_rollup")
        if previous is None:
            previous = pl.DataFrame(
                {"user_id": [], "event_count": [], "total_amount": []},
                schema={
                    "user_id": pl.String,
                    "event_count": pl.Int64,
                    "total_amount": pl.Float64,
                },
            )

        combined = pl.concat(
            [
                previous,
                df.select(
                    [
                        "user_id",
                        pl.col("event_count").cast(pl.Int64),
                        pl.col("total_amount").cast(pl.Float64),
                    ]
                ),
            ]
        )

        ctx = pl.SQLContext()
        ctx.register("combined", combined)
        rollup = ctx.execute(
            """
            SELECT
                user_id,
                SUM(event_count) AS event_count,
                SUM(total_amount) AS total_amount
            FROM combined
            GROUP BY user_id
            ORDER BY user_id
            """,
            eager=True,
        )
        if state is not None:
            state.save_parquet("user_rollup", rollup)

        rollup.write_parquet(out_dir / "cumulative.parquet")
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
    write_raw_batch(raw_dir, batch_id=1, rows=5)

    pipeline.run(once=True)

    print("batch 0 result:")
    print(pl.read_parquet(out_dir / "batch_0.parquet"))
    print("batch 1 result:")
    print(pl.read_parquet(out_dir / "batch_1.parquet"))
    print("cumulative result:")
    print(pl.read_parquet(out_dir / "cumulative.parquet"))


if __name__ == "__main__":
    main()
