import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import polars as pl

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from polars_incremental.sinks import delta as delta_sink


class TestWriteDelta(unittest.TestCase):
    def test_write_delta_lazy_collects(self) -> None:
        df = pl.DataFrame({"id": [1]})
        lf = df.lazy()
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(pl.DataFrame, "write_delta", return_value=None) as patched:
                result = delta_sink.write_delta(lf, tmpdir, mode="append")
            patched.assert_called_once()
            self.assertEqual(result, "polars")

    def test_write_delta_fallback_disabled_raises(self) -> None:
        df = pl.DataFrame({"id": [1]})
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(pl.DataFrame, "write_delta", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    delta_sink.write_delta(df, tmpdir, mode="append")
            fallback_dir = Path(tmpdir) / "_fallback_parquet"
            self.assertFalse(fallback_dir.exists())

    def test_write_delta_does_not_import_deltalake(self) -> None:
        df = pl.DataFrame({"id": [1]})
        with tempfile.TemporaryDirectory() as tmpdir:
            original_import = __import__

            def guarded_import(name, *args, **kwargs):
                if name == "deltalake":
                    raise AssertionError("deltalake import attempted")
                return original_import(name, *args, **kwargs)

            with mock.patch.object(pl.DataFrame, "write_delta", return_value=None):
                with mock.patch("builtins.__import__", side_effect=guarded_import):
                    result = delta_sink.write_delta(df, tmpdir, mode="append")
        self.assertEqual(result, "polars")


class TestApplyCdc(unittest.TestCase):
    def test_apply_cdc_invalid_mode(self) -> None:
        df = pl.DataFrame(
            {"id": [1], "value": [10], "_change_type": ["insert"], "_commit_version": [1]}
        )
        with self.assertRaises(ValueError):
            delta_sink.apply_cdc_delta(df, "/tmp/unused", keys=["id"], mode="nope")

    def test_apply_cdc_missing_key_column(self) -> None:
        df = pl.DataFrame({"value": [10], "_change_type": ["insert"]})
        with self.assertRaises(ValueError):
            delta_sink.apply_cdc_delta(df, "/tmp/unused", keys=["id"])

    def test_apply_cdc_append_only_no_existing_payload_empty(self) -> None:
        df = pl.DataFrame({"id": [1], "_change_type": ["delete"]})
        with mock.patch.object(delta_sink, "_read_delta_if_exists", return_value=None):
            with mock.patch.object(delta_sink, "write_delta") as write_delta:
                result = delta_sink.apply_cdc_delta(df, "/tmp/unused", keys=["id"], mode="append_only")
        self.assertEqual(result["action"], "noop")
        write_delta.assert_not_called()

    def test_apply_cdc_append_only_dedupes_without_commit_cols(self) -> None:
        df = pl.DataFrame(
            {
                "id": [1, 1, 2],
                "value": [10, 11, 20],
                "_change_type": ["insert", "insert", "insert"],
            }
        )
        captured: dict[str, pl.DataFrame] = {}

        def fake_write_delta(payload, *_args, **_kwargs):
            captured["payload"] = payload
            return "polars"

        with mock.patch.object(delta_sink, "_read_delta_if_exists", return_value=None):
            with mock.patch.object(delta_sink, "write_delta", side_effect=fake_write_delta):
                result = delta_sink.apply_cdc_delta(
                    df, "/tmp/unused", keys=["id"], mode="append_only"
                )
        self.assertEqual(result["action"], "append_only")
        self.assertIn("payload", captured)
        payload = captured["payload"]
        self.assertEqual(payload.height, 2)
        self.assertNotIn("_change_type", payload.columns)

    def test_apply_cdc_append_only_appends_when_existing(self) -> None:
        df = pl.DataFrame(
            {
                "id": [1, 1],
                "value": [10, 11],
                "_change_type": ["insert", "insert"],
                "_commit_version": [1, 2],
            }
        )
        existing = pl.DataFrame({"id": [99], "value": [999]})
        captured: dict[str, object] = {}

        def fake_write_delta(payload, _target, mode="append", **_kwargs):
            captured["payload"] = payload
            captured["mode"] = mode
            return "polars"

        with mock.patch.object(delta_sink, "_read_delta_if_exists", return_value=existing):
            with mock.patch.object(delta_sink, "write_delta", side_effect=fake_write_delta):
                result = delta_sink.apply_cdc_delta(
                    df, "/tmp/unused", keys=["id"], mode="append_only"
                )
        self.assertEqual(result["action"], "append_only")
        self.assertEqual(captured.get("mode"), "append")
        payload = captured["payload"]
        assert isinstance(payload, pl.DataFrame)
        self.assertEqual(payload.height, 1)
        self.assertEqual(payload["id"].to_list(), [1])
        self.assertEqual(payload["value"].to_list(), [11])

    def test_apply_cdc_uses_in_memory_merge_even_with_deltalake_present(self) -> None:
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "value": [10, None, 30],
                "_change_type": ["update_postimage", "delete", "insert"],
                "_commit_version": [1, 2, 3],
            }
        )
        existing = pl.DataFrame({"id": [1, 2], "value": [5, 20]})
        captured: dict[str, object] = {}

        def fake_write_delta(payload, _target, mode="overwrite", **_kwargs):
            captured["payload"] = payload
            captured["mode"] = mode
            return "polars"

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "delta"
            (target / "_delta_log").mkdir(parents=True, exist_ok=True)
            with mock.patch.object(delta_sink, "_read_delta_if_exists", return_value=existing) as read_delta:
                with mock.patch.object(delta_sink, "write_delta", side_effect=fake_write_delta) as write_delta:
                    result = delta_sink.apply_cdc_delta(df, target, keys=["id"], mode="merge")

        self.assertEqual(result["action"], "merge")
        self.assertEqual(result["rows_in"], df.height)
        self.assertEqual(captured.get("mode"), "overwrite")
        self.assertTrue(read_delta.called)
        self.assertTrue(write_delta.called)
        payload = captured["payload"]
        assert isinstance(payload, pl.DataFrame)
        out = payload.sort("id")
        self.assertEqual(out["id"].to_list(), [1, 3])
        self.assertEqual(out["value"].to_list(), [10, 30])

    def test_read_delta_if_exists_returns_none_without_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertIsNone(delta_sink._read_delta_if_exists(tmpdir))
