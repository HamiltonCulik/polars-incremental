from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl

import polars_incremental as pli


def main() -> None:
    base = Path("data/delta_schema_evolution_run")
    raw_dir = base / "raw"
    checkpoint_dir = base / "checkpoint"
    delta_dir = base / "delta"

    if base.exists():
        shutil.rmtree(base)

    raw_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame({"id": [1, 2], "amount": [10, 20]}).write_parquet(
        raw_dir / "batch-0000.parquet"
    )

    def reader(files):
        return pl.read_parquet(files)

    def writer(df, _batch=None):
        df.write_delta(
            delta_dir,
            mode="append",
            delta_write_options={"schema_mode": "merge"},
        )

    pipeline = pli.Pipeline(
        source=pli.FilesSource(path=raw_dir, file_format="parquet"),
        checkpoint_dir=checkpoint_dir,
        reader=reader,
        writer=writer,
        schema_evolution=pli.SchemaEvolution(
            mode="coerce",
            schema={"amount": "Float64"},
        ),
    )
    pipeline.run(once=True)

    pl.DataFrame({"id": [3], "amount": [10.5], "note": ["late"]}).write_parquet(
        raw_dir / "batch-0001.parquet"
    )

    pipeline.run(once=True)

    table = pl.read_delta(str(delta_dir))
    schema = table.schema
    print(table)
    print(schema)

    if "note" not in schema:
        raise AssertionError("Expected new column 'note' to be in the Delta schema")

    amount_dtype = schema.get("amount")
    if amount_dtype != pl.Float64:
        raise AssertionError(
            f"Expected 'amount' to be Float64 after evolution, got {amount_dtype}"
        )

    print("schema evolution verified")


if __name__ == "__main__":
    main()
