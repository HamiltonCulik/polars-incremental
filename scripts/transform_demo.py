from __future__ import annotations

from pathlib import Path

import polars as pl

import polars_incremental as pli


def main() -> None:
    base = Path("data/transform_demo")
    raw_dir = base / "raw"
    checkpoint_dir = base / "checkpoint_v2"
    out_dir = base / "out"

    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]}).write_parquet(
        raw_dir / "part-0000.parquet"
    )

    def reader(files):
        return pl.scan_parquet(files)

    def transform(lf):
        return lf.filter(pl.col("id") >= 2).with_columns(
            (pl.col("value") * 3).alias("value3")
        )

    def writer(lf, batch_id=None):
        lf.sink_parquet(out_dir / f"batch_{batch_id}.parquet")

    pipeline = pli.Pipeline(
        source=pli.FilesSource(path=raw_dir, file_format="parquet"),
        checkpoint_dir=checkpoint_dir,
        reader=reader,
        transform=transform,
        writer=writer,
    )
    result = pipeline.run(once=True)

    print(f"batches: {result.batches}")
    print(pl.read_parquet(out_dir / "batch_0.parquet"))


if __name__ == "__main__":
    main()
