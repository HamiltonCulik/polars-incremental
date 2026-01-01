from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import polars_incremental as pli
from polars_incremental.sinks import write_delta


def main() -> None:
    base_dir = Path("data/lazy_delta_write_demo")
    raw_dir = base_dir / "raw"
    checkpoint_dir = base_dir / "checkpoint"
    delta_dir = base_dir / "delta"

    if base_dir.exists():
        shutil.rmtree(base_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame({"id": [1, 2], "value": [10, 20]}).write_parquet(
        raw_dir / "part-0000.parquet"
    )
    pl.DataFrame({"id": [3], "value": [30]}).write_parquet(
        raw_dir / "part-0001.parquet"
    )

    def reader(files: list[str]):
        return pl.scan_parquet(files)

    def writer(lf: pl.LazyFrame, batch_id: int | None = None):
        mode = "overwrite" if batch_id == 0 else "append"
        backend = write_delta(
            lf,
            delta_dir,
            mode=mode,
            collect_kwargs={"engine": "streaming"},
        )
        print(f"write_delta backend={backend} batch_id={batch_id}")

    pipeline = pli.Pipeline(
        source=pli.FilesSource(
            path=raw_dir,
            file_format="parquet",
            max_files_per_trigger=1,
        ),
        checkpoint_dir=checkpoint_dir,
        reader=reader,
        writer=writer,
    )
    result = pipeline.run(once=False, max_batches=2, sleep=0.0)

    df = pl.read_delta(str(delta_dir))
    assert result.batches == 2, f"expected 2 batches, got {result.batches}"
    assert df.height == 3, f"expected 3 rows, got {df.height}"

    print("ok: lazy delta write")


if __name__ == "__main__":
    main()
