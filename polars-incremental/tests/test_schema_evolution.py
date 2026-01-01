import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from polars_incremental.errors import SchemaEvolutionError
from polars_incremental.schema import SchemaEvolution, normalize_schema_input
from polars_incremental.sources.file import FileSource


class TestSchemaEvolution(unittest.TestCase):
    def test_infer_schema_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir.mkdir(parents=True, exist_ok=True)

            df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
            df.write_parquet(input_dir / "part-0000.parquet")

            source = FileSource(input_dir, checkpoint_dir, "parquet", options={})
            batch = source.plan_batch()
            assert batch is not None
            df = source.read_batch(batch)
            SchemaEvolution(mode="add_new_columns").apply(df, source.checkpoint)

            schema = source.checkpoint.get_schema()
            self.assertIsNotNone(schema)
            assert schema is not None
            self.assertEqual([col["name"] for col in schema], ["a", "b"])

    def test_strict_rejects_new_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir.mkdir(parents=True, exist_ok=True)

            df = pl.DataFrame({"a": [1], "b": ["x"]})
            df.write_parquet(input_dir / "part-0000.parquet")

            source = FileSource(input_dir, checkpoint_dir, "parquet", options={})
            source.checkpoint.set_schema(normalize_schema_input({"a": "Int64"}))

            batch = source.plan_batch()
            assert batch is not None
            with self.assertRaises(SchemaEvolutionError):
                SchemaEvolution(mode="strict").apply(source.read_batch(batch), source.checkpoint)

    def test_add_new_columns_updates_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir.mkdir(parents=True, exist_ok=True)

            df = pl.DataFrame({"a": [1], "b": ["x"]})
            df.write_parquet(input_dir / "part-0000.parquet")

            source = FileSource(input_dir, checkpoint_dir, "parquet", options={})
            source.checkpoint.set_schema(normalize_schema_input({"a": "Int64"}))

            batch = source.plan_batch()
            assert batch is not None
            SchemaEvolution(mode="add_new_columns").apply(
                source.read_batch(batch), source.checkpoint
            )

            schema = source.checkpoint.get_schema()
            self.assertIsNotNone(schema)
            assert schema is not None
            self.assertEqual([col["name"] for col in schema], ["a", "b"])

    def test_coerce_with_rescue(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir.mkdir(parents=True, exist_ok=True)

            df = pl.DataFrame({"a": ["1", "x"]})
            df.write_parquet(input_dir / "part-0000.parquet")

            source = FileSource(input_dir, checkpoint_dir, "parquet", options={})
            source.checkpoint.set_schema(normalize_schema_input({"a": "Int64"}))

            batch = source.plan_batch()
            assert batch is not None
            result = SchemaEvolution(
                mode="coerce",
                rescue_mode="column",
                rescue_column="_rescued",
            ).apply(source.read_batch(batch), source.checkpoint)

            self.assertEqual(result["a"].to_list(), [1, None])
            rescued = result.select(pl.col("_rescued").struct.field("a")).to_series().to_list()
            self.assertEqual(rescued, [None, "x"])

    def test_schema_evolution_across_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir.mkdir(parents=True, exist_ok=True)

            pl.DataFrame({"a": [1]}).write_parquet(input_dir / "part-0000.parquet")

            source = FileSource(input_dir, checkpoint_dir, "parquet", options={})

            batch1 = source.plan_batch()
            assert batch1 is not None
            SchemaEvolution(mode="add_new_columns").apply(
                source.read_batch(batch1), source.checkpoint
            )
            source.commit_batch(batch1)

            pl.DataFrame({"a": [2], "b": ["x"]}).write_parquet(input_dir / "part-0001.parquet")

            batch2 = source.plan_batch()
            assert batch2 is not None
            result = SchemaEvolution(mode="add_new_columns").apply(
                source.read_batch(batch2), source.checkpoint
            )
            source.commit_batch(batch2)

            self.assertEqual(result.columns, ["a", "b"])
            schema = source.checkpoint.get_schema()
            self.assertIsNotNone(schema)
            assert schema is not None
            self.assertEqual([col["name"] for col in schema], ["a", "b"])

    def test_type_widen_promotes_int_to_float(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir.mkdir(parents=True, exist_ok=True)

            df = pl.DataFrame({"a": [1.5, 2.5]})
            df.write_parquet(input_dir / "part-0000.parquet")

            source = FileSource(input_dir, checkpoint_dir, "parquet", options={})
            source.checkpoint.set_schema(normalize_schema_input({"a": "Int64"}))

            batch = source.plan_batch()
            assert batch is not None
            result = SchemaEvolution(mode="type_widen").apply(
                source.read_batch(batch), source.checkpoint
            )

            self.assertEqual(result.schema["a"], pl.Float64)
            schema = source.checkpoint.get_schema()
            self.assertIsNotNone(schema)
            assert schema is not None
            self.assertEqual(schema[0]["dtype"], "Float64")

    def test_type_widen_promotes_int_to_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir.mkdir(parents=True, exist_ok=True)

            df = pl.DataFrame({"a": ["1", "2"]})
            df.write_parquet(input_dir / "part-0000.parquet")

            source = FileSource(input_dir, checkpoint_dir, "parquet", options={})
            source.checkpoint.set_schema(normalize_schema_input({"a": "Int64"}))

            batch = source.plan_batch()
            assert batch is not None
            result = SchemaEvolution(mode="type_widen").apply(
                source.read_batch(batch), source.checkpoint
            )

            self.assertEqual(result.schema["a"], pl.Utf8)
            schema = source.checkpoint.get_schema()
            self.assertIsNotNone(schema)
            assert schema is not None
            self.assertEqual(schema[0]["dtype"], "String")

    def test_type_widen_rejects_unsupported_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir.mkdir(parents=True, exist_ok=True)

            df = pl.DataFrame({"a": [1, 2]})
            df.write_parquet(input_dir / "part-0000.parquet")

            source = FileSource(input_dir, checkpoint_dir, "parquet", options={})
            source.checkpoint.set_schema(normalize_schema_input({"a": "Date"}))

            batch = source.plan_batch()
            assert batch is not None
            with self.assertRaises(SchemaEvolutionError):
                SchemaEvolution(mode="type_widen").apply(
                    source.read_batch(batch), source.checkpoint
                )
