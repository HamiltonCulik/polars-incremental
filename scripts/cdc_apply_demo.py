from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl

import polars_incremental as pli


def main() -> None:
    base = Path("data/cdc_apply_demo")
    if base.exists():
        shutil.rmtree(base)

    raw_dir = base / "cdc_raw"
    checkpoint_dir = base / "checkpoint"
    target_dir = base / "delta"

    raw_dir.mkdir(parents=True, exist_ok=True)

    # Seed target table
    pl.DataFrame({"id": [1, 2], "value": [10, 20]}).write_delta(
        target_dir, mode="overwrite"
    )

    # Batch 0: update id=2, delete id=1, insert id=3
    pl.DataFrame(
        {
            "id": [2, 1, 3],
            "value": [25, None, 30],
            "_change_type": ["update_postimage", "delete", "insert"],
            "_commit_version": [1, 1, 1],
        }
    ).write_parquet(raw_dir / "batch-0000.parquet")

    # Batch 1: update id=2 again (newer version), insert id=4
    pl.DataFrame(
        {
            "id": [2, 4],
            "value": [26, 40],
            "_change_type": ["update_postimage", "insert"],
            "_commit_version": [2, 2],
        }
    ).write_parquet(raw_dir / "batch-0001.parquet")

    def reader(files):
        return pl.read_parquet(files)

    def writer(df, batch_id=None):
        return pli.apply_cdc_delta(df, target_dir, keys=["id"])

    pipeline = pli.Pipeline(
        source=pli.FilesSource(path=raw_dir, file_format="parquet"),
        checkpoint_dir=checkpoint_dir,
        reader=reader,
        writer=writer,
    )

    pipeline.run(once=True)

    result = pl.read_delta(str(target_dir)).sort("id")
    print(result)


if __name__ == "__main__":
    main()
