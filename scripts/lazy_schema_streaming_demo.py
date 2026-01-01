from __future__ import annotations

import json
import shutil
from pathlib import Path

import polars as pl
import polars_incremental as pli
from polars_incremental.sinks import write_parquet_batch


def main() -> None:
    base_dir = Path("data/lazy_schema_streaming_demo")
    raw_dir = base_dir / "raw"
    checkpoint_dir = base_dir / "checkpoint"
    out_dir = base_dir / "out"

    if base_dir.exists():
        shutil.rmtree(base_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame({"id": [1, 2], "value": [10, 20]}).write_parquet(
        raw_dir / "part-0000.parquet"
    )
    pl.DataFrame({"id": [3], "value": [30], "extra": ["x"]}).write_parquet(
        raw_dir / "part-0001.parquet"
    )

    def reader(files: list[str]):
        return pl.scan_parquet(files)

    def writer(lf: pl.LazyFrame, batch_id: int | None = None):
        return write_parquet_batch(lf, out_dir, batch_id or 0)

    pipeline = pli.Pipeline(
        source=pli.FilesSource(
            path=raw_dir,
            file_format="parquet",
            max_files_per_trigger=1,
        ),
        checkpoint_dir=checkpoint_dir,
        reader=reader,
        writer=writer,
        schema_evolution=pli.SchemaEvolution(mode="add_new_columns"),
    )
    result = pipeline.run(once=False, max_batches=2, sleep=0.0)

    batch0 = pl.read_parquet(out_dir / "batch_0.parquet")
    batch1 = pl.read_parquet(out_dir / "batch_1.parquet")

    assert result.batches == 2, f"expected 2 batches, got {result.batches}"
    assert "extra" not in batch0.columns, "batch_0 unexpectedly has extra column"
    assert "extra" in batch1.columns, "batch_1 missing extra column"

    metadata = json.loads((checkpoint_dir / "metadata.json").read_text())
    schema_names = [entry["name"] for entry in metadata.get("schema", [])]
    assert "extra" in schema_names, "checkpoint schema did not record extra column"

    print("ok: lazy schema evolution + parquet sink")


if __name__ == "__main__":
    main()
