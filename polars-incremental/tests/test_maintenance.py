import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

import polars_incremental as pli


class TestMaintenance(unittest.TestCase):
    def test_cleanup_checkpoint_keep_last_n(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            offsets = checkpoint_dir / "offsets"
            commits = checkpoint_dir / "commits"
            offsets.mkdir(parents=True, exist_ok=True)
            commits.mkdir(parents=True, exist_ok=True)

            for i in range(5):
                (offsets / f"{i}.json").write_text("{}")
                (commits / f"{i}.json").write_text("{}")

            result = pli.cleanup_checkpoint(checkpoint_dir, keep_last_n=2)
            self.assertEqual(result.removed_offsets, 3)
            self.assertEqual(result.removed_commits, 3)
            self.assertEqual(result.kept_offsets, 2)
            self.assertEqual(result.kept_commits, 2)

    def test_cleanup_checkpoint_older_than(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            offsets = checkpoint_dir / "offsets"
            offsets.mkdir(parents=True, exist_ok=True)
            old_file = offsets / "0.json"
            new_file = offsets / "1.json"
            old_file.write_text("{}")
            new_file.write_text("{}")

            old_ts = 1000.0
            new_ts = 2000.0

            import os

            os.utime(old_file, (old_ts, old_ts))
            os.utime(new_file, (new_ts, new_ts))

            with mock.patch("polars_incremental.maintenance.time.time", return_value=2500.0):
                result = pli.cleanup_checkpoint(
                    checkpoint_dir, older_than_seconds=1000.0
                )
            self.assertEqual(result.removed_offsets, 1)
            self.assertEqual(result.kept_offsets, 1)

    def test_vacuum_delta_table_calls_deltalake(self) -> None:
        fake_table = mock.Mock()
        with mock.patch("polars_incremental.maintenance._get_delta_table", return_value=fake_table):
            pli.vacuum_delta_table("/tmp/table", retention_hours=24.0, dry_run=True)
        fake_table.vacuum.assert_called_once()

    def test_vacuum_delta_table_handles_legacy_signature(self) -> None:
        calls: list[dict] = []

        def vacuum(**kwargs):
            calls.append(dict(kwargs))
            if "enforce_retention_duration" in kwargs:
                raise TypeError("legacy signature")
            return {"ok": True}

        fake_table = mock.Mock()
        fake_table.vacuum = vacuum
        with mock.patch("polars_incremental.maintenance._get_delta_table", return_value=fake_table):
            result = pli.vacuum_delta_table(
                "/tmp/table", retention_hours=1.0, dry_run=True, enforce_retention=False
            )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(len(calls), 2)

    def test_optimize_delta_table_compact(self) -> None:
        fake_table = mock.Mock()
        optimizer = mock.Mock()
        fake_table.optimize = optimizer
        with mock.patch("polars_incremental.maintenance._get_delta_table", return_value=fake_table):
            pli.optimize_delta_table("/tmp/table", mode="compact")
        optimizer.compact.assert_called_once()

    def test_optimize_delta_table_z_order(self) -> None:
        fake_table = mock.Mock()
        optimizer = mock.Mock()
        fake_table.optimize = optimizer
        with mock.patch("polars_incremental.maintenance._get_delta_table", return_value=fake_table):
            pli.optimize_delta_table("/tmp/table", mode="z_order", z_order_columns=["a", "b"])
        optimizer.z_order.assert_called_once_with(["a", "b"])

    def test_optimize_delta_table_z_order_requires_columns(self) -> None:
        fake_table = mock.Mock()
        optimizer = mock.Mock()
        fake_table.optimize = optimizer
        with mock.patch("polars_incremental.maintenance._get_delta_table", return_value=fake_table):
            with self.assertRaises(ValueError):
                pli.optimize_delta_table("/tmp/table", mode="z_order")

    def test_optimize_delta_table_missing_optimize(self) -> None:
        fake_table = mock.Mock(spec=[])
        with mock.patch("polars_incremental.maintenance._get_delta_table", return_value=fake_table):
            with self.assertRaises(RuntimeError):
                pli.optimize_delta_table("/tmp/table", mode="compact")

    def test_reset_checkpoint_start_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            metadata = checkpoint_dir / "metadata.json"
            metadata.write_text('{"start_offset": {"mode": "latest"}, "schema": []}')

            previous = pli.reset_checkpoint_start_offset(checkpoint_dir)
            self.assertEqual(previous, {"mode": "latest"})
            payload = metadata.read_text()
            self.assertIn('"schema"', payload)
            self.assertNotIn("start_offset", payload)

    def test_reset_checkpoint_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            metadata = checkpoint_dir / "metadata.json"
            metadata.write_text('{"start_offset": {"mode": "latest"}, "schema": [{"name": "id"}]}')

            previous = pli.reset_checkpoint_schema(checkpoint_dir)
            self.assertEqual(previous, [{"name": "id"}])
            payload = metadata.read_text()
            self.assertIn('"start_offset"', payload)
            self.assertNotIn('"schema"', payload)

    def test_cleanup_snapshot_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            snapshot_dir = checkpoint_dir / "snapshot_cache" / "snapshots"
            delta_dir = checkpoint_dir / "snapshot_cache" / "deltas"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            delta_dir.mkdir(parents=True, exist_ok=True)

            for version in range(3):
                (snapshot_dir / f"{version:020d}.json").write_text("{}")
                (delta_dir / f"{version:020d}.json").write_text("{}")

            result = pli.cleanup_snapshot_cache(
                checkpoint_dir, keep_snapshots=1, keep_deltas_since_snapshot=0
            )

            self.assertEqual(result.kept_offsets, 1)
            self.assertEqual(result.kept_commits, 1)
            self.assertEqual(len(list(snapshot_dir.glob("*.json"))), 1)
            self.assertEqual(len(list(delta_dir.glob("*.json"))), 1)

    def test_truncate_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            offsets = checkpoint_dir / "offsets"
            commits = checkpoint_dir / "commits"
            offsets.mkdir(parents=True, exist_ok=True)
            commits.mkdir(parents=True, exist_ok=True)

            for i in range(5):
                (offsets / f"{i}.json").write_text("{}")
                (commits / f"{i}.json").write_text("{}")

            result = pli.truncate_checkpoint(checkpoint_dir, keep_through_batch_id=2)
            self.assertEqual(result.removed_offsets, 2)
            self.assertEqual(result.removed_commits, 2)
            self.assertEqual(result.kept_offsets, 3)
            self.assertEqual(result.kept_commits, 3)

    def test_inspect_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            offsets = checkpoint_dir / "offsets"
            commits = checkpoint_dir / "commits"
            offsets.mkdir(parents=True, exist_ok=True)
            commits.mkdir(parents=True, exist_ok=True)

            (offsets / "0.json").write_text("{}")
            (offsets / "1.json").write_text("{}")
            (commits / "0.json").write_text("{}")
            (checkpoint_dir / "metadata.json").write_text('{"start_offset": {"mode": "latest"}}')

            info = pli.inspect_checkpoint(checkpoint_dir)
            self.assertEqual(info.offsets, 2)
            self.assertEqual(info.commits, 1)
            self.assertEqual(info.latest_offset, 1)
            self.assertEqual(info.latest_commit, 0)
            self.assertEqual(info.pending, 1)
            self.assertEqual(info.start_offset, {"mode": "latest"})
            self.assertEqual(info.snapshot_cache_snapshots, 0)
            self.assertEqual(info.snapshot_cache_deltas, 0)
