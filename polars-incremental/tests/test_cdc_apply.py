import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from polars_incremental import apply_cdc_delta


class TestCdcApply(unittest.TestCase):
    def test_apply_cdc_merge_updates_and_deletes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "delta"

            pl.DataFrame({"id": [1, 2], "value": [10, 20]}).write_delta(
                target, mode="overwrite"
            )

            cdc = pl.DataFrame(
                {
                    "id": [2, 1, 3],
                    "value": [25, None, 30],
                    "_change_type": ["update_postimage", "delete", "insert"],
                    "_commit_version": [1, 1, 1],
                }
            )

            apply_cdc_delta(cdc, target, keys=["id"])
            out = pl.read_delta(str(target)).sort("id")
            self.assertEqual(out["id"].to_list(), [2, 3])
            self.assertEqual(out["value"].to_list(), [25, 30])

    def test_apply_cdc_latest_delete_wins_over_upsert(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "delta"

            pl.DataFrame({"id": [1], "value": [10]}).write_delta(
                target, mode="overwrite"
            )

            cdc = pl.DataFrame(
                {
                    "id": [1, 1],
                    "value": [99, None],
                    "_change_type": ["update_postimage", "delete"],
                    "_commit_version": [1, 2],
                }
            )

            apply_cdc_delta(cdc, target, keys=["id"])
            out = pl.read_delta(str(target))
            self.assertEqual(out.height, 0)

    def test_apply_cdc_latest_change_wins_without_commit_cols(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "delta"

            pl.DataFrame({"id": [1], "value": [10]}).write_delta(
                target, mode="overwrite"
            )

            cdc = pl.DataFrame(
                {
                    "id": [1, 1],
                    "value": [99, None],
                    "_change_type": ["update_postimage", "delete"],
                }
            )

            apply_cdc_delta(cdc, target, keys=["id"])
            out = pl.read_delta(str(target))
            self.assertEqual(out.height, 0)

    def test_apply_cdc_change_type_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "delta"

            pl.DataFrame({"id": [1, 2], "value": [10, 20]}).write_delta(
                target, mode="overwrite"
            )

            cdc = pl.DataFrame(
                {
                    "id": [2, 1, 3],
                    "value": [25, None, 30],
                    "_change_type": ["U", "D", "I"],
                    "_commit_version": [1, 2, 3],
                }
            )

            apply_cdc_delta(
                cdc,
                target,
                keys=["id"],
                change_type_map={"I": "insert", "U": "update_postimage", "D": "delete"},
            )
            out = pl.read_delta(str(target)).sort("id")
            self.assertEqual(out["id"].to_list(), [2, 3])
            self.assertEqual(out["value"].to_list(), [25, 30])

    def test_apply_cdc_merge_handles_deletes_without_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "delta"

            cdc = pl.DataFrame(
                {
                    "id": [1, 1, 2],
                    "value": [10, None, 20],
                    "_change_type": ["insert", "delete", "insert"],
                    "_commit_version": [0, 1, 1],
                }
            )

            apply_cdc_delta(cdc, target, keys=["id"])
            out = pl.read_delta(str(target)).sort("id")
            self.assertEqual(out["id"].to_list(), [2])
            self.assertEqual(out["value"].to_list(), [20])

    def test_apply_cdc_dedupes_latest_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "delta"

            pl.DataFrame({"id": [1], "value": [5]}).write_delta(
                target, mode="overwrite"
            )

            cdc = pl.DataFrame(
                {
                    "id": [1, 1],
                    "value": [10, 11],
                    "_change_type": ["update_postimage", "update_postimage"],
                    "_commit_version": [1, 2],
                }
            )

            apply_cdc_delta(cdc, target, keys=["id"], dedupe_by_latest_commit=True)
            out = pl.read_delta(str(target))
            self.assertEqual(out["value"].to_list(), [11])

    def test_apply_cdc_dedupes_by_commit_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "delta"

            pl.DataFrame({"id": [1], "value": [5]}).write_delta(
                target, mode="overwrite"
            )

            cdc = pl.DataFrame(
                {
                    "id": [1, 1],
                    "value": [10, 11],
                    "_change_type": ["update_postimage", "update_postimage"],
                    "_commit_timestamp": [1000, 2000],
                }
            )

            apply_cdc_delta(cdc, target, keys=["id"], dedupe_by_latest_commit=True)
            out = pl.read_delta(str(target))
            self.assertEqual(out["value"].to_list(), [11])

    def test_apply_cdc_append_only_ignores_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "delta"

            pl.DataFrame({"id": [1], "value": [10]}).write_delta(
                target, mode="overwrite"
            )

            cdc = pl.DataFrame(
                {
                    "id": [1, 1, 2],
                    "value": [99, None, 20],
                    "_change_type": ["update_postimage", "delete", "insert"],
                }
            )

            apply_cdc_delta(cdc, target, keys=["id"], mode="append_only")
            out = pl.read_delta(str(target)).sort("id")
            self.assertEqual(out["id"].to_list(), [1, 2])
            self.assertEqual(out["value"].to_list(), [10, 20])

    def test_apply_cdc_requires_change_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "delta"
            df = pl.DataFrame({"id": [1], "value": [10]})
            with self.assertRaises(ValueError):
                apply_cdc_delta(df, target, keys=["id"])
