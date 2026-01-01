import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from polars_incremental.source import AutoSource, DeltaSource, FilesSource
from polars_incremental.sources.base import SourceSpec


class TestSourceConfig(unittest.TestCase):
    def test_files_source_to_spec_compacts_options(self) -> None:
        source = FilesSource(
            path=Path("/tmp/data"),
            file_format="parquet",
            pattern=None,
            recursive=True,
            max_files_per_trigger=10,
            max_bytes_per_trigger=None,
            max_file_age=3600.0,
            start_offset="latest",
            start_timestamp=None,
            allow_overwrites=True,
            clean_source="delete",
        )
        spec = source.to_spec()
        self.assertIsInstance(spec, SourceSpec)
        self.assertEqual(spec.format, "files")
        self.assertIn("file_format", spec.options)
        self.assertNotIn("pattern", spec.options)
        self.assertEqual(spec.options["recursive"], True)
        self.assertEqual(spec.options["max_file_age"], 3600.0)
        self.assertEqual(spec.options["clean_source"], "delete")

    def test_delta_source_to_spec_compacts_options(self) -> None:
        source = DeltaSource(
            path=Path("/tmp/table"),
            start_offset="latest",
            starting_version=None,
            starting_timestamp=None,
            max_files_per_trigger=100,
            max_bytes_per_trigger=None,
            ignore_deletes=True,
            ignore_changes=False,
            read_change_feed=True,
        )
        spec = source.to_spec()
        self.assertEqual(spec.format, "delta")
        self.assertEqual(spec.options["start_offset"], "latest")
        self.assertNotIn("starting_version", spec.options)
        self.assertEqual(spec.options["ignore_deletes"], True)

    def test_auto_source_to_spec(self) -> None:
        source = AutoSource(path=Path("/tmp/data"), file_format="csv", pattern="*.csv")
        spec = source.to_spec()
        self.assertEqual(spec.format, "auto")
        self.assertEqual(spec.options["file_format"], "csv")
        self.assertEqual(spec.options["pattern"], "*.csv")
