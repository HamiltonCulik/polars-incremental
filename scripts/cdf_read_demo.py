from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
from deltalake import DeltaTable

import polars_incremental as pli


def main() -> None:
    base = Path("data/cdf_demo")
    source = base / "source_table"
    sink = base / "sink_table"

    if base.exists():
        shutil.rmtree(base)
    source.mkdir(parents=True, exist_ok=True)

    # Create table with CDF enabled.
    df = pl.DataFrame({"id": [1, 2], "value": [10, 20]})
    df.write_delta(
        source,
        mode="overwrite",
        delta_write_options={"configuration": {"delta.enableChangeDataFeed": "true"}},
    )

    dt = DeltaTable(str(source))

    # Update + delete to generate CDF entries.
    dt.update(predicate="id = 2", updates={"value": "25"})
    dt.delete("id = 1")

    # Insert another row.
    insert_df = pl.DataFrame({"id": [3], "value": [30]})
    insert_df.write_delta(source, mode="append")

    def reader(files, batch=None):
        if batch is not None:
            return pli.read_cdf_batch(batch)
        return pl.read_parquet(files)

    def writer(df, batch_id=None):
        print("CDF rows:")
        print(
            df.select(
                [
                    "id",
                    "value",
                    "_change_type",
                    "_commit_version",
                    "_commit_timestamp",
                ]
            ).sort("_commit_version")
        )
        return pli.apply_cdc_delta(df, sink, keys=["id"])

    pipeline = pli.Pipeline(
        source=pli.DeltaSource(
            path=source,
            read_change_feed=True,
            starting_version=0,
        ),
        checkpoint_dir=base / "checkpoint",
        reader=reader,
        writer=writer,
    )
    result = pipeline.run(once=True)
    print("batches processed:", result.batches)

    # Apply CDC changes into a sink table.
    result = {"note": "apply_cdc executed in writer"}
    print("apply_cdc result:", result)
    print("sink preview:")
    print(pl.read_delta(str(sink)).sort("id"))


if __name__ == "__main__":
    main()
