import tempfile
import unittest
from pathlib import Path

import polars as pl

from polars_incremental.state import JobState


class TestJobState(unittest.TestCase):
    def test_json_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = JobState(Path(tmpdir))
            self.assertEqual(state.load_json("missing", default={"ok": True}), {"ok": True})

            state.save_json("config", {"value": 123})
            self.assertEqual(state.load_json("config", default={}), {"value": 123})

    def test_parquet_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = JobState(Path(tmpdir))
            self.assertIsNone(state.load_parquet("missing"))

            df = pl.DataFrame({"id": [1, 2], "value": [10, 20]})
            state.save_parquet("table", df)
            loaded = state.load_parquet("table")
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.shape, (2, 2))

    def test_exists_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = JobState(Path(tmpdir))
            self.assertFalse(state.exists("foo"))

            state.save_json("foo", {"ok": True})
            self.assertTrue(state.exists("foo"))
            self.assertTrue(state.exists("foo", kind="json"))
            self.assertFalse(state.exists("foo", kind="parquet"))

            removed = state.delete("foo", kind="json")
            self.assertTrue(removed)
            self.assertFalse(state.exists("foo"))

    def test_exists_and_delete_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = JobState(Path(tmpdir))
            df = pl.DataFrame({"id": [1]})
            state.save_parquet("table", df)
            self.assertTrue(state.exists("table", kind="parquet"))

            removed = state.delete("table", kind="parquet")
            self.assertTrue(removed)
            self.assertFalse(state.exists("table", kind="parquet"))

    def test_invalid_kind_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = JobState(Path(tmpdir))
            with self.assertRaises(ValueError):
                state.exists("foo", kind="csv")
            with self.assertRaises(ValueError):
                state.delete("foo", kind="csv")
