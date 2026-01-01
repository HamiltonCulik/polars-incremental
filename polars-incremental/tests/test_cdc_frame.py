import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from polars_incremental import apply_cdc
from polars_incremental.cdc import _normalize_change_types


class TestApplyCdc(unittest.TestCase):
    def test_apply_cdc_merge_updates_and_deletes(self) -> None:
        existing = pl.DataFrame({"id": [1, 2], "value": [10, 20]})
        changes = pl.DataFrame(
            {
                "id": [2, 1, 3],
                "value": [25, None, 30],
                "_change_type": ["update_postimage", "delete", "insert"],
                "_commit_version": [1, 1, 1],
            }
        )

        out = apply_cdc(changes, existing, keys=["id"])
        out = out.sort("id")
        self.assertEqual(out["id"].to_list(), [2, 3])
        self.assertEqual(out["value"].to_list(), [25, 30])

    def test_apply_cdc_latest_delete_wins_over_upsert(self) -> None:
        existing = pl.DataFrame({"id": [1], "value": [10]})
        changes = pl.DataFrame(
            {
                "id": [1, 1],
                "value": [99, None],
                "_change_type": ["update_postimage", "delete"],
                "_commit_version": [1, 2],
            }
        )

        out = apply_cdc(changes, existing, keys=["id"])
        self.assertEqual(out.height, 0)

    def test_apply_cdc_latest_change_wins_without_commit_cols(self) -> None:
        existing = pl.DataFrame({"id": [1], "value": [10]})
        changes = pl.DataFrame(
            {
                "id": [1, 1],
                "value": [99, None],
                "_change_type": ["update_postimage", "delete"],
            }
        )

        out = apply_cdc(changes, existing, keys=["id"])
        self.assertEqual(out.height, 0)

    def test_apply_cdc_change_type_map(self) -> None:
        existing = pl.DataFrame({"id": [1, 2], "value": [10, 20]})
        changes = pl.DataFrame(
            {
                "id": [2, 1, 3],
                "value": [25, None, 30],
                "_change_type": ["U", "D", "I"],
                "_commit_version": [1, 2, 3],
            }
        )

        out = apply_cdc(
            changes,
            existing,
            keys=["id"],
            change_type_map={"I": "insert", "U": "update_postimage", "D": "delete"},
        )
        out = out.sort("id")
        self.assertEqual(out["id"].to_list(), [2, 3])
        self.assertEqual(out["value"].to_list(), [25, 30])

    def test_apply_cdc_dedupes_by_commit_timestamp(self) -> None:
        existing = pl.DataFrame({"id": [1], "value": [5]})
        changes = pl.DataFrame(
            {
                "id": [1, 1],
                "value": [10, 11],
                "_change_type": ["update_postimage", "update_postimage"],
                "_commit_timestamp": [100, 200],
            }
        )

        out = apply_cdc(changes, existing, keys=["id"], dedupe_by_latest_commit=True)
        self.assertEqual(out["value"].to_list(), [11])

    def test_apply_cdc_append_only_inserts_only(self) -> None:
        existing = pl.DataFrame({"id": [9], "value": [5]})
        changes = pl.DataFrame(
            {
                "id": [1, 1, 2, 3, 4],
                "value": [10, 11, 20, 30, 40],
                "_change_type": [
                    "insert",
                    "insert",
                    "insert",
                    "update_postimage",
                    "delete",
                ],
            }
        )

        out = apply_cdc(changes, existing, keys=["id"], mode="append_only")
        out = out.sort(["id", "value"])
        self.assertEqual(out["id"].to_list(), [1, 2, 9])
        self.assertEqual(out["value"].to_list(), [11, 20, 5])

    def test_apply_cdc_empty_changes_returns_existing(self) -> None:
        existing = pl.DataFrame({"id": [1], "value": [10]})
        changes = pl.DataFrame({"id": [], "_change_type": []})

        out = apply_cdc(changes, existing, keys=["id"])
        self.assertEqual(out.to_dicts(), existing.to_dicts())

    def test_apply_cdc_existing_none_drops_deletes(self) -> None:
        changes = pl.DataFrame(
            {
                "id": [1, 1, 2],
                "value": [10, None, 20],
                "_change_type": ["insert", "delete", "insert"],
                "_commit_version": [0, 1, 1],
            }
        )

        out = apply_cdc(
            changes,
            existing=None,
            keys=["id"],
            dedupe_by_latest_commit=False,
        )
        out = out.sort("id")
        self.assertEqual(out["id"].to_list(), [2])
        self.assertEqual(out["value"].to_list(), [20])

    def test_apply_cdc_accepts_lazy_inputs(self) -> None:
        existing = pl.DataFrame({"id": [1], "value": [10]}).lazy()
        changes = pl.DataFrame(
            {
                "id": [1, 2],
                "value": [None, 20],
                "_change_type": ["delete", "insert"],
            }
        ).lazy()

        out = apply_cdc(changes, existing, keys=["id"])
        out = out.sort("id")
        self.assertEqual(out["id"].to_list(), [2])
        self.assertEqual(out["value"].to_list(), [20])

    def test_apply_cdc_with_parquet_io(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            existing_path = base / "existing.parquet"
            changes_path = base / "changes.parquet"
            output_path = base / "output.parquet"

            pl.DataFrame({"id": [1, 2], "value": [10, 20]}).write_parquet(existing_path)
            pl.DataFrame(
                {
                    "id": [2, 1, 3],
                    "value": [25, None, 30],
                    "_change_type": ["update_postimage", "delete", "insert"],
                    "_commit_version": [1, 2, 3],
                }
            ).write_parquet(changes_path)

            existing = pl.read_parquet(existing_path)
            changes = pl.read_parquet(changes_path)
            updated = apply_cdc(changes, existing, keys=["id"])
            updated.write_parquet(output_path)

            out = pl.read_parquet(output_path).sort("id")
            self.assertEqual(out["id"].to_list(), [2, 3])
            self.assertEqual(out["value"].to_list(), [25, 30])

    def test_normalize_change_types_preserves_unmapped(self) -> None:
        df = pl.DataFrame({"_change_type": ["I", "X"]})
        out = _normalize_change_types(
            df,
            change_type_col="_change_type",
            change_type_map={"I": "insert"},
        )
        self.assertEqual(out["_change_type"].to_list(), ["insert", "X"])
