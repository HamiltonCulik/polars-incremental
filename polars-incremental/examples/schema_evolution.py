"""Example: Schema evolution + rescue.

Scenario: Incoming data may add columns or have type mismatches; use SchemaEvolution + rescue.
"""

from pathlib import Path

import polars as pl
import polars_incremental as pli

base_dir = Path("data/schema_evolution_example")
raw_dir = base_dir / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

pl.DataFrame({"a": ["1", "x"], "b": ["ok", "bad"]}).write_parquet(raw_dir / "part-0000.parquet")

def reader(files):
    return pl.read_parquet(files)

def writer_factory(out_dir: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def writer(df, batch_id=None):
        df.write_parquet(out_path / f"batch_{batch_id}.parquet")
    return writer

# Example 1: coerce + rescue (keeps job running)
pipeline = pli.Pipeline(
    source=pli.FilesSource(path=raw_dir),
    checkpoint_dir=base_dir / "checkpoints" / "schema_coerce",
    reader=reader,
    writer=writer_factory(str(base_dir / "out" / "parquet_sink_coerce")),
    schema_evolution=pli.SchemaEvolution(
        mode="coerce",
        schema={"a": "Int64"},
        rescue_mode="column",
        rescue_column="_rescued",
    ),
)

pipeline.run(once=True)

# Example 2: add new columns without coercion
pipeline = pli.Pipeline(
    source=pli.FilesSource(path=raw_dir),
    checkpoint_dir=base_dir / "checkpoints" / "schema_add",
    reader=reader,
    writer=writer_factory(str(base_dir / "out" / "parquet_sink_add")),
    schema_evolution=pli.SchemaEvolution(mode="add_new_columns"),
)

pipeline.run(once=True)
