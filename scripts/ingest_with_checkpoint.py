from __future__ import annotations

from pathlib import Path

import polars as pl

import polars_incremental as pli


def main() -> None:
    input_dir = Path("data/raw")
    checkpoint_dir = Path("data/checkpoints/raw_stream_v2")
    delta_dir = Path("data/delta/events")

    def reader(files):
        return pl.read_parquet(files)

    def writer(df, _batch=None):
        df.write_delta(delta_dir, mode="append")

    pipeline = pli.Pipeline(
        source=pli.FilesSource(
            path=input_dir,
            file_format="parquet",
            pattern="events_[0-9]*.parquet",
        ),
        checkpoint_dir=checkpoint_dir,
        reader=reader,
        writer=writer,
    )
    result = pipeline.run(once=True)

    if result.batches == 0:
        print("no new files found")
        return

    table_df = pl.read_delta(str(delta_dir))
    print("delta table preview:")
    print(table_df.head(5))


if __name__ == "__main__":
    main()
