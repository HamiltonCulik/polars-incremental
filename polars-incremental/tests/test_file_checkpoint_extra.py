import json
import os
import shutil
import sys
import tempfile
import unittest
import time
from pathlib import Path
from unittest import mock

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from polars_incremental.checkpoints.types import BatchInfo
from polars_incremental.checkpoints.file import FileStreamCheckpoint, iter_new_files
from polars_incremental.sources.file import FileSource


class TestFileCheckpointExtra(unittest.TestCase):
    def test_file_signature_missing_returns_none(self) -> None:
        checkpoint = FileStreamCheckpoint(Path(tempfile.mkdtemp()) / "checkpoint")
        self.assertIsNone(checkpoint._file_signature("/no/such/file"))

    def test_iter_new_files_returns_empty_when_no_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir = Path(tmpdir) / "checkpoint"

            files = list(iter_new_files(input_dir, checkpoint_dir))
            self.assertEqual(files, [])

    def test_list_pending_batches_skips_missing_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            offset_dir = checkpoint_dir / "offsets"
            commit_dir = checkpoint_dir / "commits"
            offset_dir.mkdir(parents=True, exist_ok=True)
            commit_dir.mkdir(parents=True, exist_ok=True)
            (offset_dir / "0.json").write_text('{"batch_id": 0, "files": [], "created_at": 0}')
            (offset_dir / "1.json").write_text('{"batch_id": 1, "files": [], "created_at": 0}')
            (offset_dir / "2.json").write_text('{"batch_id": 2, "files": [], "created_at": 0}')
            (commit_dir / "0.json").write_text("{}")

            checkpoint = FileStreamCheckpoint(checkpoint_dir)
            original_read_offset = checkpoint.read_offset

            def fake_read_offset(batch_id):
                if batch_id == 2:
                    raise FileNotFoundError
                return original_read_offset(batch_id)

            with mock.patch.object(checkpoint, "read_offset", side_effect=fake_read_offset):
                pending = checkpoint.list_pending_batches()
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].batch_id, 1)

    def test_invalid_start_offset_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = FileStreamCheckpoint(Path(tmpdir) / "checkpoint")
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            with self.assertRaises(ValueError):
                checkpoint.plan_batch(input_dir, start_offset="weird")

    def test_file_index_keeps_missing_files_without_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            checkpoint = FileStreamCheckpoint(base / "checkpoint")
            file1 = base / "file1.parquet"
            file2 = base / "file2.parquet"
            file1.write_text("data")
            batch1 = BatchInfo(batch_id=0, files=[str(file1)], created_at=0.0)
            checkpoint.commit_batch(batch1)

            file1.unlink()
            file2.write_text("data")
            batch2 = BatchInfo(batch_id=1, files=[str(file2)], created_at=1.0)
            checkpoint.commit_batch(batch2)

            merged = checkpoint._file_index()
            self.assertIn(str(file2), merged)
            self.assertIn(str(file1), merged)

    def test_file_index_updates_only_touched_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            checkpoint = FileStreamCheckpoint(base / "checkpoint")
            index_dir = base / "checkpoint" / "file_index"
            index_dir.mkdir(parents=True, exist_ok=True)

            # Seed two shards with dummy data.
            (index_dir / "aa.json").write_text(json.dumps({"a.parquet": {"mtime_ns": 1, "size": 1}}))
            (index_dir / "bb.json").write_text(json.dumps({"b.parquet": {"mtime_ns": 1, "size": 1}}))

            target = base / "target.parquet"
            target.write_text("data")
            target_shard = checkpoint._index_shard_id(str(target))

            calls: list[str] = []

            original_read = checkpoint._read_shard

            def fake_read_shard(shard_id: str):
                calls.append(shard_id)
                return original_read(shard_id)

            with mock.patch.object(checkpoint, "_read_shard", side_effect=fake_read_shard):
                checkpoint._update_file_index([str(target)])

            self.assertEqual(calls, [target_shard])

    def test_file_index_sharded_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            checkpoint = FileStreamCheckpoint(base / "checkpoint")
            file1 = base / "alpha.parquet"
            file2 = base / "bravo.parquet"
            file1.write_text("data")
            file2.write_text("data")

            batch = BatchInfo(batch_id=0, files=[str(file1), str(file2)], created_at=0.0)
            checkpoint.commit_batch(batch)

            metadata_path = base / "checkpoint" / "metadata.json"
            metadata = metadata_path.read_text()
            self.assertIn("file_index_format", metadata)
            self.assertNotIn("\"file_index\":", metadata)

            index_dir = base / "checkpoint" / "file_index"
            self.assertTrue(index_dir.exists())
            shards = list(index_dir.glob("*.json"))
            self.assertTrue(shards)

            merged = checkpoint._file_index()
            self.assertIn(str(file1), merged)
            self.assertIn(str(file2), merged)

    def test_file_index_migrates_legacy_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            checkpoint_dir = base / "checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            legacy = {
                "format_version": 1,
                "created_at": 0,
                "file_index": {
                    str(base / "old.parquet"): {"mtime_ns": 1, "size": 2},
                },
            }
            (checkpoint_dir / "metadata.json").write_text(
                json.dumps(legacy, indent=2, sort_keys=True)
            )

            checkpoint = FileStreamCheckpoint(checkpoint_dir)
            merged = checkpoint._file_index()
            self.assertIn(str(base / "old.parquet"), merged)

            metadata = (checkpoint_dir / "metadata.json").read_text()
            self.assertIn("file_index_format", metadata)
            self.assertNotIn("\"file_index\":", metadata)

    def test_plan_batch_respects_max_file_age(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir = Path(tmpdir) / "checkpoint"

            old_file = input_dir / "old.parquet"
            new_file = input_dir / "new.parquet"
            old_file.write_text("data")
            new_file.write_text("data")

            now = time.time()
            os.utime(old_file, (now - 3600, now - 3600))
            os.utime(new_file, (now, now))

            checkpoint = FileStreamCheckpoint(checkpoint_dir)
            batch = checkpoint.plan_batch(
                input_dir=input_dir,
                pattern="*.parquet",
                max_file_age=300.0,
            )
            self.assertIsNotNone(batch)
            assert batch is not None
            self.assertEqual(batch.files, [str(new_file.resolve())])

    def test_max_file_age_prunes_index_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            input_dir = base / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir = base / "checkpoint"

            old_file = input_dir / "old.parquet"
            new_file = input_dir / "new.parquet"
            old_file.write_text("data")
            now = time.time()
            os.utime(old_file, (now - 3600, now - 3600))

            checkpoint = FileStreamCheckpoint(checkpoint_dir)
            batch = checkpoint.plan_batch(input_dir=input_dir, pattern="*.parquet")
            assert batch is not None
            checkpoint.write_offset(batch)
            checkpoint.commit_batch(batch)

            new_file.write_text("data")
            os.utime(new_file, (now, now))
            batch2 = checkpoint.plan_batch(
                input_dir=input_dir,
                pattern="*.parquet",
                max_file_age=300.0,
            )
            assert batch2 is not None
            self.assertEqual(batch2.files, [str(new_file.resolve())])
            self.assertNotIn(str(old_file.resolve()), checkpoint._file_index())

    def test_max_file_age_relative_to_latest_allows_old_only_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir = Path(tmpdir) / "checkpoint"

            old_file = input_dir / "old.parquet"
            old_file.write_text("data")
            now = time.time()
            os.utime(old_file, (now - 3600, now - 3600))

            checkpoint = FileStreamCheckpoint(checkpoint_dir)
            batch = checkpoint.plan_batch(
                input_dir=input_dir,
                pattern="*.parquet",
                max_file_age=300.0,
            )
            self.assertIsNotNone(batch)
            assert batch is not None
            self.assertEqual(batch.files, [str(old_file.resolve())])

    def test_clean_source_delete_removes_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            input_dir = base / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            file_path = input_dir / "part-0000.parquet"
            file_path.write_text("data")

            source = FileSource(
                path=input_dir,
                checkpoint_dir=base / "checkpoint",
                source_format="parquet",
                options={"clean_source": "delete"},
            )
            batch = source.plan_batch()
            assert batch is not None
            source.commit_batch(batch)

            self.assertFalse(file_path.exists())
            self.assertNotIn(str(file_path.resolve()), source.checkpoint._file_index())

    def test_clean_source_delete_failure_keeps_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            input_dir = base / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            file_path = input_dir / "part-0000.parquet"
            file_path.write_text("data")

            source = FileSource(
                path=input_dir,
                checkpoint_dir=base / "checkpoint",
                source_format="parquet",
                options={"clean_source": "delete"},
            )
            batch = source.plan_batch()
            assert batch is not None

            with mock.patch("pathlib.Path.unlink", side_effect=OSError("fail")):
                source.commit_batch(batch)

            self.assertTrue(file_path.exists())
            self.assertIn(str(file_path.resolve()), source.checkpoint._file_index())

    def test_clean_source_archive_moves_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            input_dir = base / "input"
            archive_dir = base / "archive"
            input_dir.mkdir(parents=True, exist_ok=True)
            file_path = input_dir / "part-0000.parquet"
            file_path.write_text("data")

            source = FileSource(
                path=input_dir,
                checkpoint_dir=base / "checkpoint",
                source_format="parquet",
                options={
                    "clean_source": "archive",
                    "clean_source_archive_dir": str(archive_dir),
                },
            )
            batch = source.plan_batch()
            assert batch is not None
            source.commit_batch(batch)

            self.assertFalse(file_path.exists())
            archived = archive_dir / file_path.name
            self.assertTrue(archived.exists())
            self.assertNotIn(str(file_path.resolve()), source.checkpoint._file_index())

    def test_recursive_archive_dir_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            input_dir = base / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            archive_dir = input_dir / "_archive"
            archive_dir.mkdir(parents=True, exist_ok=True)

            fresh_file = input_dir / "fresh.parquet"
            archived_file = archive_dir / "archived.parquet"
            fresh_file.write_text("data")
            archived_file.write_text("data")

            source = FileSource(
                path=input_dir,
                checkpoint_dir=base / "checkpoint",
                source_format="parquet",
                options={"recursive": True, "clean_source": "archive"},
            )
            batch = source.plan_batch()
            self.assertIsNotNone(batch)
            assert batch is not None
            self.assertEqual(batch.files, [str(fresh_file.resolve())])

    def test_allow_overwrites_does_not_write_legacy_file_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            input_dir = base / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir = base / "checkpoint"

            file_path = input_dir / "file.parquet"
            file_path.write_text("data")

            checkpoint = FileStreamCheckpoint(checkpoint_dir)
            batch = checkpoint.plan_batch(input_dir=input_dir, pattern="*.parquet")
            assert batch is not None
            checkpoint.write_offset(batch)
            checkpoint.commit_batch(batch)

            index_dir = checkpoint_dir / "file_index"
            if index_dir.exists():
                shutil.rmtree(index_dir)

            batch2 = checkpoint.plan_batch(
                input_dir=input_dir,
                pattern="*.parquet",
                allow_overwrites=True,
            )
            self.assertIsNone(batch2)

            metadata = json.loads((checkpoint_dir / "metadata.json").read_text())
            self.assertNotIn("file_index", metadata)

            index_dir = checkpoint_dir / "file_index"
            shards = list(index_dir.glob("*.json"))
            self.assertTrue(shards)
            merged: dict[str, dict] = {}
            for shard in shards:
                payload = json.loads(shard.read_text())
                if isinstance(payload, dict):
                    merged.update(payload)
            self.assertIn(str(file_path.resolve()), merged)
