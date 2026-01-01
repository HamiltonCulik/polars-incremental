import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from polars_incremental import BatchInfo, FileStreamCheckpoint


def _touch_files(base: Path, names: list[str]) -> list[str]:
    base.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for name in names:
        path = base / name
        path.write_text("data")
        files.append(str(path.resolve()))
    return files


def _set_mtime(path: Path, ts: float) -> None:
    os.utime(path, (ts, ts))


class TestFileStreamCheckpoint(unittest.TestCase):
    def test_plan_batch_empty_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir = Path(tmpdir) / "input"
            checkpoint = FileStreamCheckpoint(checkpoint_dir)

            batch = checkpoint.plan_batch(input_dir, pattern="*.parquet")

            self.assertIsNone(batch)

    def test_plan_batch_and_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir = Path(tmpdir) / "input"
            _touch_files(input_dir, ["b.parquet", "a.parquet", "c.parquet"])

            checkpoint = FileStreamCheckpoint(checkpoint_dir)
            with patch("polars_incremental.checkpoints.file.time.time", return_value=123.0):
                batch = checkpoint.plan_batch(input_dir, pattern="*.parquet")
                self.assertIsNotNone(batch)
                assert batch is not None
                self.assertEqual(batch.batch_id, 0)
                self.assertEqual(batch.files, sorted(batch.files))
                checkpoint.write_offset(batch)
                checkpoint.commit_batch(batch)

            self.assertEqual(checkpoint.latest_commit_batch_id(), 0)
            self.assertIsNone(checkpoint.plan_batch(input_dir, pattern="*.parquet"))

    def test_recover_uncommitted_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir = Path(tmpdir) / "input"
            _touch_files(input_dir, ["a.parquet", "b.parquet"])

            checkpoint = FileStreamCheckpoint(checkpoint_dir)
            with patch("polars_incremental.checkpoints.file.time.time", return_value=111.0):
                batch = checkpoint.plan_batch(input_dir, pattern="*.parquet")
                assert batch is not None
                checkpoint.write_offset(batch)

            # New instance should recover the same uncommitted batch
            recovered = FileStreamCheckpoint(checkpoint_dir).plan_batch(
                input_dir, pattern="*.parquet"
            )
            self.assertIsNotNone(recovered)
            assert recovered is not None
            self.assertEqual(recovered.batch_id, batch.batch_id)
            self.assertEqual(recovered.files, batch.files)

    def test_max_files_limits_and_advances(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir = Path(tmpdir) / "input"
            files = _touch_files(
                input_dir, ["a.parquet", "b.parquet", "c.parquet", "d.parquet", "e.parquet"]
            )

            checkpoint = FileStreamCheckpoint(checkpoint_dir)
            batch1 = checkpoint.plan_batch(input_dir, max_files=2)
            assert batch1 is not None
            self.assertEqual(batch1.files, files[:2])
            checkpoint.write_offset(batch1)
            checkpoint.commit_batch(batch1)

            batch2 = checkpoint.plan_batch(input_dir, max_files=2)
            assert batch2 is not None
            self.assertEqual(batch2.files, files[2:4])

    def test_max_bytes_limits_and_advances(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir(parents=True, exist_ok=True)

            files: list[str] = []
            for name, size in [("a.parquet", 3), ("b.parquet", 5), ("c.parquet", 5)]:
                path = input_dir / name
                path.write_bytes(b"x" * size)
                files.append(str(path.resolve()))

            checkpoint = FileStreamCheckpoint(checkpoint_dir)
            batch1 = checkpoint.plan_batch(input_dir, max_bytes=7)
            assert batch1 is not None
            self.assertEqual(batch1.files, files[:1])
            checkpoint.write_offset(batch1)
            checkpoint.commit_batch(batch1)

            batch2 = checkpoint.plan_batch(input_dir, max_bytes=7)
            assert batch2 is not None
            self.assertEqual(batch2.files, files[1:2])

    def test_list_pending_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir = Path(tmpdir) / "input"
            files = _touch_files(input_dir, ["a.parquet", "b.parquet", "c.parquet"])

            checkpoint = FileStreamCheckpoint(checkpoint_dir)
            batch0 = BatchInfo(batch_id=0, files=[files[0]], created_at=1.0)
            batch1 = BatchInfo(batch_id=1, files=[files[1], files[2]], created_at=2.0)
            checkpoint.write_offset(batch0)
            checkpoint.write_offset(batch1)
            checkpoint.commit_batch(batch0)

            pending = checkpoint.list_pending_batches()
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].batch_id, 1)

    def test_start_offset_latest_skips_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir = Path(tmpdir) / "input"
            paths = _touch_files(input_dir, ["a.parquet", "b.parquet"])
            for path in paths:
                _set_mtime(Path(path), 900.0)

            checkpoint = FileStreamCheckpoint(checkpoint_dir)
            with patch("polars_incremental.checkpoints.file.time.time", return_value=1000.0):
                batch = checkpoint.plan_batch(input_dir, start_offset="latest")
            self.assertIsNone(batch)

            new_path = Path(input_dir) / "c.parquet"
            new_path.write_text("data")
            _set_mtime(new_path, 1002.0)

            batch2 = checkpoint.plan_batch(input_dir)
            self.assertIsNotNone(batch2)
            assert batch2 is not None
            self.assertEqual(batch2.files, [str(new_path.resolve())])

    def test_start_offset_timestamp_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir = Path(tmpdir) / "input"
            paths = _touch_files(input_dir, ["a.parquet", "b.parquet", "c.parquet"])
            _set_mtime(Path(paths[0]), 1000.0)
            _set_mtime(Path(paths[1]), 1010.0)
            _set_mtime(Path(paths[2]), 1020.0)

            checkpoint = FileStreamCheckpoint(checkpoint_dir)
            batch = checkpoint.plan_batch(input_dir, start_timestamp=1010.0)
            self.assertIsNotNone(batch)
            assert batch is not None
            self.assertEqual(
                batch.files,
                [
                    str(Path(paths[1]).resolve()),
                    str(Path(paths[2]).resolve()),
                ],
            )

    def test_start_offset_persists_across_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir = Path(tmpdir) / "input"
            paths = _touch_files(input_dir, ["old.parquet"])
            _set_mtime(Path(paths[0]), 900.0)

            checkpoint = FileStreamCheckpoint(checkpoint_dir)
            with patch("polars_incremental.checkpoints.file.time.time", return_value=1000.0):
                batch = checkpoint.plan_batch(input_dir, start_offset="latest")
            self.assertIsNone(batch)

            # Change requested start_offset, but persisted metadata should keep "latest".
            checkpoint2 = FileStreamCheckpoint(checkpoint_dir)
            with self.assertLogs("polars_incremental", level="WARNING"):
                batch2 = checkpoint2.plan_batch(input_dir, start_offset="earliest")
            self.assertIsNone(batch2)

            new_path = Path(input_dir) / "new.parquet"
            new_path.write_text("data")
            _set_mtime(new_path, 1005.0)

            with self.assertLogs("polars_incremental", level="WARNING"):
                batch3 = checkpoint2.plan_batch(input_dir, start_offset="earliest")
            self.assertIsNotNone(batch3)
            assert batch3 is not None
            self.assertEqual(batch3.files, [str(new_path.resolve())])

    def test_start_offset_warning_when_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir = Path(tmpdir) / "input"
            _touch_files(input_dir, ["a.parquet"])

            checkpoint = FileStreamCheckpoint(checkpoint_dir)
            batch = checkpoint.plan_batch(input_dir, start_offset="earliest")
            assert batch is not None
            checkpoint.write_offset(batch)
            checkpoint.commit_batch(batch)

            with self.assertLogs("polars_incremental", level="WARNING") as captured:
                checkpoint.plan_batch(input_dir, start_offset="latest")
            self.assertTrue(
                any("start_offset" in message for message in captured.output),
                "expected a warning when start_offset is ignored",
            )

    def test_allow_overwrites_reprocesses_modified_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir = Path(tmpdir) / "input"
            paths = _touch_files(input_dir, ["a.parquet"])
            _set_mtime(Path(paths[0]), 1000.0)

            checkpoint = FileStreamCheckpoint(checkpoint_dir)
            batch = checkpoint.plan_batch(input_dir, allow_overwrites=True)
            assert batch is not None
            checkpoint.write_offset(batch)
            checkpoint.commit_batch(batch)

            # modify file
            _set_mtime(Path(paths[0]), 2000.0)

            batch2 = checkpoint.plan_batch(input_dir, allow_overwrites=True)
            self.assertIsNotNone(batch2)
            assert batch2 is not None
            self.assertEqual(batch2.files, [str(Path(paths[0]).resolve())])

    def test_allow_overwrites_skips_unchanged_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir = Path(tmpdir) / "input"
            paths = _touch_files(input_dir, ["a.parquet"])
            _set_mtime(Path(paths[0]), 1000.0)

            checkpoint = FileStreamCheckpoint(checkpoint_dir)
            batch = checkpoint.plan_batch(input_dir, allow_overwrites=True)
            assert batch is not None
            checkpoint.write_offset(batch)
            checkpoint.commit_batch(batch)

            batch2 = checkpoint.plan_batch(input_dir, allow_overwrites=True)
            self.assertIsNone(batch2)
