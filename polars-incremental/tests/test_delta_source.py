import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

import json

from polars_incremental.checkpoints.types import DeltaBatch, DeltaFileEntry, DeltaOffset
from polars_incremental.schema import SchemaEvolution, normalize_schema_input
from polars_incremental.sources.delta import DeltaSource, read_cdf_batch


class TestDeltaSource(unittest.TestCase):
    def test_read_cdf_batch_empty(self) -> None:
        batch = DeltaBatch(
            batch_id=0,
            offset=DeltaOffset(table_id="t1", version=0, index=0, is_initial_snapshot=False),
            files=[],
            created_at=0.0,
        )
        out = read_cdf_batch(batch)
        self.assertTrue(out.is_empty())

    def test_read_cdf_batch_without_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            file_path = base / "part-0000.parquet"
            pl.DataFrame({"id": [1], "value": [10]}).write_parquet(file_path)

            batch = DeltaBatch(
                batch_id=0,
                offset=DeltaOffset(table_id="t1", version=0, index=0, is_initial_snapshot=False),
                files=[str(file_path)],
                created_at=0.0,
            )

            out = read_cdf_batch(batch)
            self.assertEqual(out["id"].to_list(), [1])

    def test_read_cdf_batch_injects_metadata_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            file_path = base / "cdf-0000.parquet"
            pl.DataFrame({"id": [1], "value": [10]}).write_parquet(file_path)

            batch = DeltaBatch(
                batch_id=0,
                offset=DeltaOffset(table_id="t1", version=1, index=0, is_initial_snapshot=False),
                files=[str(file_path)],
                created_at=0.0,
                file_entries=[
                    DeltaFileEntry(
                        path=str(file_path),
                        commit_version=1,
                        commit_timestamp=123,
                        change_type="insert",
                        size=10,
                    )
                ],
            )

            out = read_cdf_batch(batch)
            self.assertIn("_change_type", out.columns)
            self.assertIn("_commit_version", out.columns)
            self.assertIn("_commit_timestamp", out.columns)
            self.assertEqual(out["_change_type"].to_list(), ["insert"])
            self.assertEqual(out["_commit_version"].to_list(), [1])
            self.assertEqual(out["_commit_timestamp"].to_list(), [123])

    def test_read_cdf_batch_empty_entries(self) -> None:
        batch = DeltaBatch(
            batch_id=0,
            offset=DeltaOffset(table_id="t1", version=0, index=0, is_initial_snapshot=False),
            files=["/tmp/unused.parquet"],
            created_at=0.0,
            file_entries=[],
        )
        out = read_cdf_batch(batch)
        self.assertTrue(out.is_empty())

    def test_read_batch_applies_schema_and_rescue(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            table_path.mkdir(parents=True, exist_ok=True)
            df_path = table_path / "part-0000.parquet"

            pl.DataFrame({"a": ["1", "x"]}).write_parquet(df_path)

            source = DeltaSource(
                path=table_path,
                checkpoint_dir=checkpoint_dir,
                options={},
            )
            source.checkpoint.set_schema(normalize_schema_input({"a": "Int64"}))

            batch = DeltaBatch(
                batch_id=0,
                offset=DeltaOffset(table_id="t1", version=0, index=0, is_initial_snapshot=False),
                files=[str(df_path)],
                created_at=0.0,
            )
            out = SchemaEvolution(
                mode="coerce",
                rescue_mode="column",
                rescue_column="_rescued",
            ).apply(source.read_batch(batch), source.checkpoint)

            self.assertEqual(out["a"].to_list(), [1, None])
            rescued = out.select(pl.col("_rescued").struct.field("a")).to_series().to_list()
            self.assertEqual(rescued, [None, "x"])

    def test_plan_batch_writes_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"
            table_path.mkdir(parents=True, exist_ok=True)
            _write_log(
                log_dir,
                0,
                [
                    {"commitInfo": {"timestamp": 1000}},
                    {"add": {"path": "part-0000.parquet", "size": 10}},
                ],
            )
            source = DeltaSource(
                path=table_path,
                checkpoint_dir=Path(tmpdir) / "checkpoint",
                options={},
            )
            batch = source.plan_batch()
            self.assertIsNotNone(batch)
            assert batch is not None
            offset_file = source.checkpoint.offset_dir / "0.json"
            self.assertTrue(offset_file.exists())

    def test_plan_batch_returns_none_without_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = DeltaSource(
                path=Path(tmpdir) / "table",
                checkpoint_dir=Path(tmpdir) / "checkpoint",
                options={},
            )
            self.assertIsNone(source.plan_batch())

    def test_read_batch_empty_files_returns_empty(self) -> None:
        source = DeltaSource(path="/tmp/table", checkpoint_dir="/tmp/checkpoint", options={})
        batch = DeltaBatch(
            batch_id=0,
            offset=DeltaOffset(table_id="t1", version=0, index=0, is_initial_snapshot=False),
            files=[],
            created_at=0.0,
        )
        out = source.read_batch(batch)
        self.assertTrue(out.is_empty())

    def test_read_batch_change_feed_applies_schema_evolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            table_path.mkdir(parents=True, exist_ok=True)
            df_path = table_path / "part-0000.parquet"
            pl.DataFrame({"a": [1]}).write_parquet(df_path)

            source = DeltaSource(
                path=table_path,
                checkpoint_dir=checkpoint_dir,
                options={"read_change_feed": True},
            )
            batch = DeltaBatch(
                batch_id=0,
                offset=DeltaOffset(table_id="t1", version=0, index=0, is_initial_snapshot=False),
                files=[str(df_path)],
                created_at=0.0,
                file_entries=[
                    DeltaFileEntry(
                        path=str(df_path),
                        commit_version=0,
                        commit_timestamp=None,
                        change_type="insert",
                        size=10,
                    )
                ],
            )
            schema_evolution = SchemaEvolution(mode="add_new_columns")
            with unittest.mock.patch.object(source.checkpoint, "set_schema") as set_schema:
                schema_evolution.apply(source.read_batch(batch), source.checkpoint)
            set_schema.assert_called_once()

    def test_commit_batch_calls_checkpoint(self) -> None:
        source = DeltaSource(path="/tmp/table", checkpoint_dir="/tmp/checkpoint", options={})
        batch = DeltaBatch(
            batch_id=0,
            offset=DeltaOffset(table_id="t1", version=0, index=0, is_initial_snapshot=False),
            files=[],
            created_at=0.0,
        )
        with unittest.mock.patch.object(source.checkpoint, "commit_batch") as commit:
            source.commit_batch(batch, metadata={"ok": True})
        commit.assert_called_once()


def _write_log(log_dir: Path, version: int, actions: list[dict]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{version:020d}.json"
    with path.open("w", encoding="utf-8") as handle:
        for action in actions:
            handle.write(json.dumps(action))
            handle.write("\n")
