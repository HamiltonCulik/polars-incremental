import json
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from polars_incremental import DeltaTableCheckpoint, read_cdf_batch, ChangeDataFeedError


def _write_log(log_dir: Path, version: int, actions: list[dict]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{version:020d}.json"
    with path.open("w", encoding="utf-8") as handle:
        for action in actions:
            handle.write(json.dumps(action))
            handle.write("\n")


class TestDeltaChangeDataFeed(unittest.TestCase):
    def test_cdf_reads_cdc_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"
            cdc_dir = table_path / "_change_data"
            cdc_dir.mkdir(parents=True, exist_ok=True)

            df = pl.DataFrame({"id": [1], "_change_type": ["delete"]})
            cdc_file = cdc_dir / "cdc-0000.parquet"
            df.write_parquet(cdc_file)

            _write_log(
                log_dir,
                0,
                [
                    {"commitInfo": {"timestamp": 123456}},
                    {"metaData": {"id": "table-1"}},
                    {"cdc": {"path": "_change_data/cdc-0000.parquet", "size": 10}},
                ],
            )

            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            batch = checkpoint.plan_batch(
                table_path, read_change_feed=True, starting_version=0
            )
            self.assertIsNotNone(batch)
            assert batch is not None
            self.assertEqual(batch.files, [str(cdc_file.resolve())])
            self.assertIsNotNone(batch.file_entries)

            result = read_cdf_batch(batch)
            self.assertIn("_change_type", result.columns)
            self.assertIn("_commit_version", result.columns)
            self.assertIn("_commit_timestamp", result.columns)
            self.assertEqual(result.select("_commit_version").to_series()[0], 0)
            self.assertEqual(result.select("_commit_timestamp").to_series()[0], 123456)

    def test_cdf_falls_back_to_inserts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"
            data_dir = table_path / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            df = pl.DataFrame({"id": [1]})
            data_file = data_dir / "part-0000.parquet"
            df.write_parquet(data_file)

            _write_log(
                log_dir,
                0,
                [
                    {"commitInfo": {"timestamp": 2000}},
                    {"metaData": {"id": "table-1"}},
                    {"add": {"path": "data/part-0000.parquet", "size": 10}},
                ],
            )

            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            batch = checkpoint.plan_batch(
                table_path, read_change_feed=True, starting_version=0
            )
            self.assertIsNotNone(batch)
            assert batch is not None
            result = read_cdf_batch(batch)
            self.assertIn("_change_type", result.columns)
            self.assertEqual(result.select("_change_type").to_series()[0], "insert")

    def test_cdf_delete_without_cdc_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"

            _write_log(
                log_dir,
                0,
                [
                    {"commitInfo": {"timestamp": 3000}},
                    {"metaData": {"id": "table-1"}},
                    {"remove": {"path": "data/part-0000.parquet"}},
                ],
            )

            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            with self.assertRaises(ChangeDataFeedError):
                checkpoint.plan_batch(table_path, read_change_feed=True, starting_version=0)
