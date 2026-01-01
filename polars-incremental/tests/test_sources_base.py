import tempfile
import unittest
from pathlib import Path

from polars_incremental.errors import UnsupportedFormatError
from polars_incremental.sources.base import (
    SourceSpec,
    _infer_format_from_pattern,
    _infer_source_format,
    _normalize_file_format,
)


class TestSourceInference(unittest.TestCase):
    def test_normalize_file_format(self) -> None:
        self.assertEqual(_normalize_file_format("CSV"), "csv")
        self.assertEqual(_normalize_file_format("jsonl"), "ndjson")
        self.assertEqual(_normalize_file_format("xlsx"), "excel")
        self.assertIsNone(_normalize_file_format(123))
        self.assertIsNone(_normalize_file_format("unknown"))

    def test_infer_format_from_pattern(self) -> None:
        self.assertEqual(_infer_format_from_pattern("data.csv"), "csv")
        self.assertEqual(_infer_format_from_pattern("data.jsonl"), "ndjson")
        self.assertEqual(_infer_format_from_pattern("data.parq"), "parquet")
        self.assertIsNone(_infer_format_from_pattern(123))

    def test_infer_source_format_delta_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "table"
            (base / "_delta_log").mkdir(parents=True, exist_ok=True)
            fmt = _infer_source_format(base, {})
            self.assertEqual(fmt, "delta")

    def test_infer_source_format_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "data.json"
            base.write_text("{}")
            fmt = _infer_source_format(base, {})
            self.assertEqual(fmt, "json")

    def test_source_spec_rejects_unknown_format(self) -> None:
        spec = SourceSpec(format="weird", path="data", options={})
        with self.assertRaises(UnsupportedFormatError):
            spec.with_checkpoint("checkpoint")
