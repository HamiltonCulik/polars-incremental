import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from polars_incremental.checkpoints.delta import DeltaTableCheckpoint
from polars_incremental.errors import ChangeDataFeedError


def _write_log(log_dir: Path, version: int, actions: list[dict]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{version:020d}.json"
    with path.open("w", encoding="utf-8") as handle:
        for action in actions:
            handle.write(json.dumps(action))
            handle.write("\n")


def _touch_files(base: Path, names: list[str]) -> None:
    base.mkdir(parents=True, exist_ok=True)
    for name in names:
        (base / name).write_text("data")


class TestDeltaCheckpointExtra(unittest.TestCase):
    def test_parse_starting_timestamp(self) -> None:
        checkpoint = DeltaTableCheckpoint(Path(tempfile.mkdtemp()) / "checkpoint")
        expected = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        self.assertEqual(checkpoint._parse_starting_timestamp("2023-01-01T00:00:00Z"), expected)
        self.assertEqual(checkpoint._parse_starting_timestamp("2023-01-01"), expected)

    def test_build_snapshot_batch_respects_max_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"
            _touch_files(table_path, ["part-0000.parquet", "part-0001.parquet"])

            _write_log(
                log_dir,
                0,
                [
                    {"commitInfo": {"timestamp": 1000}},
                    {"add": {"path": "part-0000.parquet", "size": 100}},
                    {"add": {"path": "part-0001.parquet", "size": 100}},
                ],
            )

            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            batch = checkpoint._build_snapshot_batch(
                table_path=table_path,
                version=0,
                start_index=0,
                max_files=None,
                max_bytes=10,
            )
            self.assertIsNotNone(batch)
            assert batch is not None
            self.assertEqual(len(batch.files), 1)

    def test_build_log_batch_advances_without_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"
            _write_log(
                log_dir,
                0,
                [
                    {"commitInfo": {"timestamp": 1000}},
                    {"remove": {"path": "part-0000.parquet"}},
                ],
            )
            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            batch = checkpoint._build_log_batch(
                table_path=table_path,
                start_version=0,
                start_index=-1,
                max_files=None,
                max_bytes=None,
                ignore_deletes=True,
                ignore_changes=False,
            )
            self.assertIsNotNone(batch)
            assert batch is not None
            self.assertEqual(batch.files, [])

    def test_cdf_entries_raise_on_remove_without_cdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"
            _write_log(
                log_dir,
                0,
                [
                    {"commitInfo": {"timestamp": 1000}},
                    {"remove": {"path": "part-0000.parquet"}},
                ],
            )
            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            with self.assertRaises(ChangeDataFeedError):
                checkpoint._cdf_entries_for_version(table_path, 0)

    def test_cdf_entries_use_cdc_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"
            _touch_files(table_path, ["cdf-0000.parquet"])
            _write_log(
                log_dir,
                0,
                [
                    {"commitInfo": {"timestamp": 1000}},
                    {"cdc": {"path": "cdf-0000.parquet", "size": 10}},
                ],
            )
            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            entries, last_index, used_cdf = checkpoint._cdf_entries_for_version(table_path, 0)
            self.assertTrue(used_cdf)
            self.assertEqual(last_index, 0)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].commit_timestamp, 1000)

    def test_warns_when_start_offset_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            checkpoint._metadata["start_offset"] = {"mode": "latest", "version": 3}

            with self.assertLogs("polars_incremental", level="WARNING") as captured:
                checkpoint._warn_if_start_offset_ignored(
                    start_offset="earliest",
                    starting_version=None,
                    starting_timestamp=None,
                )

            self.assertTrue(
                any("start_offset" in message for message in captured.output),
                "expected a warning when start_offset is ignored",
            )

    def test_read_offset_uses_file_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            checkpoint = DeltaTableCheckpoint(checkpoint_dir)
            offset_path = checkpoint.offset_dir / "0.json"
            payload = {
                "batch_id": 0,
                "created_at": 0.0,
                "table_id": "t1",
                "version": 1,
                "index": 0,
                "is_initial_snapshot": False,
                "file_entries": [
                    {
                        "path": "/tmp/file.parquet",
                        "commit_version": 1,
                        "commit_timestamp": 123,
                        "change_type": "insert",
                        "size": 10,
                    }
                ],
            }
            offset_path.write_text(json.dumps(payload))
            batch = checkpoint.read_offset(0)
            self.assertEqual(batch.files, ["/tmp/file.parquet"])

    def test_resolve_start_offset_config_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"
            _write_log(
                log_dir,
                0,
                [
                    {"commitInfo": {"timestamp": 1000}},
                    {"add": {"path": "part-0000.parquet", "size": 10}},
                ],
            )
            _write_log(
                log_dir,
                1,
                [
                    {"commitInfo": {"timestamp": 2000}},
                    {"add": {"path": "part-0001.parquet", "size": 10}},
                ],
            )
            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint_latest")
            latest = checkpoint._resolve_start_offset_config(
                table_path=table_path,
                start_offset="latest",
                starting_version=None,
                starting_timestamp=None,
            )
            self.assertEqual(latest, {"mode": "latest", "version": 1})

            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint_earliest")
            earliest = checkpoint._resolve_start_offset_config(
                table_path=table_path,
                start_offset="earliest",
                starting_version=None,
                starting_timestamp=None,
            )
            self.assertEqual(earliest, {"mode": "version", "version": 0})

    def test_resolve_starting_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"
            _write_log(
                log_dir,
                0,
                [{"commitInfo": {"timestamp": 1000}}],
            )
            _write_log(
                log_dir,
                1,
                [{"commitInfo": {"timestamp": 2000}}],
            )
            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            config = checkpoint._resolve_start_offset_config(
                table_path=table_path,
                start_offset=None,
                starting_version=None,
                starting_timestamp="1970-01-01T00:00:01Z",
            )
            self.assertEqual(config, {"mode": "version", "version": 0, "timestamp": "1970-01-01T00:00:01Z"})
