import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from polars_incremental import DeltaTableCheckpoint


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


class TestDeltaTableCheckpoint(unittest.TestCase):
    def test_initial_snapshot_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"
            _touch_files(table_path, ["part-0000.parquet", "part-0001.parquet", "part-0002.parquet"])

            _write_log(
                log_dir,
                0,
                [
                    {"commitInfo": {"timestamp": 1000}},
                    {"metaData": {"id": "table-1"}},
                    {"add": {"path": "part-0000.parquet", "size": 10}},
                    {"add": {"path": "part-0001.parquet", "size": 20}},
                ],
            )
            _write_log(
                log_dir,
                1,
                [
                    {"commitInfo": {"timestamp": 2000}},
                    {"add": {"path": "part-0002.parquet", "size": 30}},
                    {"remove": {"path": "part-0000.parquet"}},
                ],
            )

            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            batch1 = checkpoint.plan_batch(table_path, max_files=1)
            self.assertIsNotNone(batch1)
            assert batch1 is not None
            self.assertTrue(batch1.offset.is_initial_snapshot)
            self.assertEqual(batch1.offset.version, 1)
            self.assertEqual(len(batch1.files), 1)

            checkpoint.write_offset(batch1)
            checkpoint.commit_batch(batch1)

            batch2 = checkpoint.plan_batch(table_path, max_files=1)
            self.assertIsNotNone(batch2)
            assert batch2 is not None
            self.assertTrue(batch2.offset.is_initial_snapshot)
            self.assertEqual(batch2.offset.version, 1)
            self.assertEqual(len(batch2.files), 1)
            self.assertNotEqual(batch1.files, batch2.files)

    def test_starting_version_skips_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"
            _touch_files(table_path, ["part-0000.parquet", "part-0001.parquet"])

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
                    {"add": {"path": "part-0001.parquet", "size": 20}},
                ],
            )

            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            batch = checkpoint.plan_batch(table_path, starting_version=1)
            self.assertIsNotNone(batch)
            assert batch is not None
            self.assertFalse(batch.offset.is_initial_snapshot)
            self.assertEqual(batch.offset.version, 1)
            self.assertEqual(batch.files, [str((table_path / "part-0001.parquet").resolve())])

    def test_ignore_changes_allows_add_remove(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"
            _touch_files(table_path, ["part-0000.parquet", "part-0001.parquet"])

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
                    {"add": {"path": "part-0001.parquet", "size": 20}},
                    {"remove": {"path": "part-0000.parquet"}},
                ],
            )

            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            with self.assertRaises(RuntimeError):
                checkpoint.plan_batch(table_path, starting_version=1)

            batch = checkpoint.plan_batch(table_path, starting_version=1, ignore_changes=True)
            self.assertIsNotNone(batch)
            assert batch is not None
            self.assertEqual(len(batch.files), 1)

    def test_ignore_deletes_advances(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"
            _touch_files(table_path, ["part-0000.parquet"])

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
                    {"remove": {"path": "part-0000.parquet"}},
                ],
            )

            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            batch = checkpoint.plan_batch(
                table_path, starting_version=1, ignore_deletes=True
            )
            self.assertIsNotNone(batch)
            assert batch is not None
            self.assertEqual(batch.files, [])
            checkpoint.write_offset(batch)
            checkpoint.commit_batch(batch)

            self.assertIsNone(
                checkpoint.plan_batch(table_path, starting_version=1, ignore_deletes=True)
            )

    def test_ignore_changes_advances_on_remove_only_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"
            _touch_files(table_path, ["part-0000.parquet"])

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
                    {"remove": {"path": "part-0000.parquet"}},
                ],
            )

            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            batch = checkpoint.plan_batch(
                table_path, starting_version=1, ignore_changes=True
            )
            self.assertIsNotNone(batch)
            assert batch is not None
            self.assertEqual(batch.files, [])
            checkpoint.write_offset(batch)
            checkpoint.commit_batch(batch)

            self.assertIsNone(
                checkpoint.plan_batch(
                    table_path, starting_version=1, ignore_changes=True
                )
            )

    def test_datachange_false_advances_without_ignore_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"
            _touch_files(table_path, ["part-0000.parquet", "part-0001.parquet"])

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
                    {"add": {"path": "part-0001.parquet", "size": 20, "dataChange": False}},
                    {"remove": {"path": "part-0000.parquet", "dataChange": False}},
                ],
            )

            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            batch = checkpoint.plan_batch(table_path, starting_version=1)
            self.assertIsNotNone(batch)
            assert batch is not None
            self.assertEqual(batch.files, [])
            self.assertEqual(batch.offset.version, 1)

            checkpoint.write_offset(batch)
            checkpoint.commit_batch(batch)
            self.assertIsNone(checkpoint.plan_batch(table_path, starting_version=1))

    def test_cdf_datachange_false_advances(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"
            _touch_files(table_path, ["part-0000.parquet", "part-0001.parquet"])

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
                    {"add": {"path": "part-0001.parquet", "size": 20, "dataChange": False}},
                    {"remove": {"path": "part-0000.parquet", "dataChange": False}},
                ],
            )

            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            batch = checkpoint.plan_batch(
                table_path, starting_version=1, read_change_feed=True
            )
            self.assertIsNotNone(batch)
            assert batch is not None
            self.assertEqual(batch.files, [])
            self.assertEqual(batch.offset.version, 1)

            checkpoint.write_offset(batch)
            checkpoint.commit_batch(batch)
            self.assertIsNone(
                checkpoint.plan_batch(
                    table_path, starting_version=1, read_change_feed=True
                )
            )

    def test_mixed_datachange_add_remove_without_ignore_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"
            _touch_files(table_path, ["part-0000.parquet", "part-0001.parquet"])

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
                    {"add": {"path": "part-0001.parquet", "size": 20, "dataChange": True}},
                    {"remove": {"path": "part-0000.parquet", "dataChange": False}},
                ],
            )

            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            batch = checkpoint.plan_batch(table_path, starting_version=1)
            self.assertIsNotNone(batch)
            assert batch is not None
            self.assertEqual(
                batch.files, [str((table_path / "part-0001.parquet").resolve())]
            )
            self.assertEqual(batch.offset.version, 1)

    def test_start_offset_latest_skips_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"

            _write_log(
                log_dir,
                0,
                [
                    {"commitInfo": {"timestamp": 1000}},
                    {"metaData": {"id": "table-1"}},
                    {"add": {"path": "part-0000.parquet", "size": 10}},
                ],
            )

            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            batch = checkpoint.plan_batch(table_path, start_offset="latest")
            self.assertIsNone(batch)

            _write_log(
                log_dir,
                1,
                [
                    {"commitInfo": {"timestamp": 2000}},
                    {"add": {"path": "part-0001.parquet", "size": 20}},
                ],
            )

            batch2 = checkpoint.plan_batch(table_path)
            self.assertIsNotNone(batch2)
            assert batch2 is not None
            self.assertFalse(batch2.offset.is_initial_snapshot)
            self.assertEqual(batch2.offset.version, 1)

    def test_table_id_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"
            _touch_files(table_path, ["part-0000.parquet"])

            _write_log(
                log_dir,
                0,
                [
                    {"commitInfo": {"timestamp": 1000}},
                    {"metaData": {"id": "table-1"}},
                    {"add": {"path": "part-0000.parquet", "size": 10}},
                ],
            )

            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            batch = checkpoint.plan_batch(table_path)
            self.assertIsNotNone(batch)
            assert batch is not None
            checkpoint.write_offset(batch)
            checkpoint.commit_batch(batch)

            _write_log(
                log_dir,
                0,
                [
                    {"commitInfo": {"timestamp": 1000}},
                    {"metaData": {"id": "table-2"}},
                    {"add": {"path": "part-0000.parquet", "size": 10}},
                ],
            )

            with self.assertRaises(RuntimeError):
                checkpoint.plan_batch(table_path)

    def test_snapshot_state_uses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            calls: list[int] = []

            def fake_iter(_table_path, version):
                calls.append(version)
                return iter(())

            with mock.patch.object(checkpoint, "_iter_log_actions", side_effect=fake_iter):
                checkpoint._snapshot_state("/tmp/table", 1)
                checkpoint._snapshot_state("/tmp/table", 1)
                checkpoint._snapshot_state("/tmp/table", 2)

            self.assertEqual(calls, [0, 1, 2])

    def test_snapshot_state_uses_persisted_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"
            table_path.mkdir(parents=True, exist_ok=True)

            _write_log(
                log_dir,
                0,
                [
                    {"commitInfo": {"timestamp": 1000}},
                    {"metaData": {"id": "table-1"}},
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
            _write_log(
                log_dir,
                2,
                [
                    {"commitInfo": {"timestamp": 3000}},
                    {"add": {"path": "part-0002.parquet", "size": 10}},
                ],
            )

            checkpoint1 = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            checkpoint1._snapshot_state(table_path, 1)

            checkpoint2 = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            calls: list[int] = []

            def fake_iter(_table_path, version):
                calls.append(version)
                return iter(())

            with mock.patch.object(checkpoint2, "_iter_log_actions", side_effect=fake_iter):
                checkpoint2._snapshot_state(table_path, 2)

            self.assertEqual(calls, [2])

    def test_snapshot_state_writes_snapshot_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"
            table_path.mkdir(parents=True, exist_ok=True)

            _write_log(
                log_dir,
                0,
                [
                    {"commitInfo": {"timestamp": 1000}},
                    {"metaData": {"id": "table-1"}},
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

            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            checkpoint._snapshot_state(table_path, 1)

            snapshot_path = (
                checkpoint.checkpoint_dir
                / "snapshot_cache"
                / "snapshots"
                / f"{1:020d}.json"
            )
            delta_path = (
                checkpoint.checkpoint_dir
                / "snapshot_cache"
                / "deltas"
                / f"{1:020d}.json"
            )
            self.assertTrue(snapshot_path.exists())
            self.assertFalse(delta_path.exists())

            metadata = json.loads((checkpoint.checkpoint_dir / "metadata.json").read_text())
            cache = metadata.get("snapshot_cache", {})
            self.assertEqual(cache.get("version"), 1)
            self.assertNotIn("active", cache)

    def test_snapshot_state_uses_delta_cache_without_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"
            table_path.mkdir(parents=True, exist_ok=True)

            _write_log(
                log_dir,
                0,
                [
                    {"commitInfo": {"timestamp": 1000}},
                    {"metaData": {"id": "table-1"}},
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
            _write_log(
                log_dir,
                2,
                [
                    {"commitInfo": {"timestamp": 3000}},
                    {"add": {"path": "part-0002.parquet", "size": 10}},
                ],
            )

            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            checkpoint._snapshot_state(table_path, 1)
            checkpoint._snapshot_state(table_path, 2)

            checkpoint2 = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            with mock.patch.object(
                checkpoint2, "_iter_log_actions", side_effect=AssertionError("log read")
            ):
                checkpoint2._snapshot_state(table_path, 2)

    def test_plan_batch_creates_snapshot_cache_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"
            table_path.mkdir(parents=True, exist_ok=True)

            _write_log(
                log_dir,
                0,
                [
                    {"commitInfo": {"timestamp": 1000}},
                    {"metaData": {"id": "table-1"}},
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

            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            batch = checkpoint.plan_batch(table_path)
            self.assertIsNotNone(batch)
            snapshot_path = (
                checkpoint.checkpoint_dir
                / "snapshot_cache"
                / "snapshots"
                / f"{1:020d}.json"
            )
            self.assertTrue(snapshot_path.exists())

    def test_legacy_snapshot_cache_migrates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "format_version": 1,
                "source": "delta",
                "created_at": 0,
                "snapshot_cache": {
                    "version": 1,
                    "table_id": "table-1",
                    "active": {"part-0000.parquet": 10},
                },
            }
            (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata))

            checkpoint = DeltaTableCheckpoint(checkpoint_dir)
            snapshot_path = (
                checkpoint_dir / "snapshot_cache" / "snapshots" / f"{1:020d}.json"
            )
            self.assertTrue(snapshot_path.exists())

            payload = json.loads((checkpoint_dir / "metadata.json").read_text())
            cache = payload.get("snapshot_cache", {})
            self.assertEqual(cache.get("version"), 1)
            self.assertEqual(cache.get("table_id"), "table-1")
            self.assertNotIn("active", cache)

    def test_snapshot_cache_pruning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "table"
            log_dir = table_path / "_delta_log"
            table_path.mkdir(parents=True, exist_ok=True)

            for version in range(3):
                _write_log(
                    log_dir,
                    version,
                    [
                        {"commitInfo": {"timestamp": 1000 + version}},
                        {"metaData": {"id": "table-1"}},
                        {"add": {"path": f"part-{version:04d}.parquet", "size": 10}},
                    ],
                )

            checkpoint = DeltaTableCheckpoint(Path(tmpdir) / "checkpoint")
            checkpoint._SNAPSHOT_EVERY = 1
            checkpoint._MAX_SNAPSHOTS = 1

            checkpoint._snapshot_state(table_path, 0)
            checkpoint._snapshot_state(table_path, 1)
            checkpoint._snapshot_state(table_path, 2)

            snapshot_dir = checkpoint.checkpoint_dir / "snapshot_cache" / "snapshots"
            snapshots = sorted(p.name for p in snapshot_dir.glob("*.json"))
            self.assertEqual(snapshots, [f"{2:020d}.json"])

            delta_dir = checkpoint.checkpoint_dir / "snapshot_cache" / "deltas"
            deltas = list(delta_dir.glob("*.json")) if delta_dir.exists() else []
            self.assertEqual(deltas, [])
