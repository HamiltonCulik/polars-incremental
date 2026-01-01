import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

import polars_incremental as pli


class TestCatalog(unittest.TestCase):
    def test_catalog_scan_and_sink(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            pl.DataFrame({"id": [1, 2]}).write_parquet(input_dir / "part-0000.parquet")

            catalog = pli.LocalCatalog(
                {
                    "datasets": {
                        "raw_events": {
                            "format": "parquet",
                            "path": str(input_dir),
                        },
                    "events_out": {
                        "format": "parquet",
                        "path": str(output_dir),
                    },
                    }
                }
            )

            source = catalog.get_source("raw_events")
            sink = catalog.resolve("events_out")

            def reader(files):
                return pl.scan_parquet(files).with_columns((pl.col("id") * 10).alias("id2"))

            def writer(lf, batch):
                df = lf.collect()
                out_path = output_dir / f"batch_{batch.batch_id}.parquet"
                df.write_parquet(out_path)
                return {"path": str(out_path), "rows": df.height}

            pipeline = pli.Pipeline(
                source=source,
                checkpoint_dir=checkpoint_dir,
                reader=reader,
                writer=writer,
            )
            result = pipeline.run(once=True)
            self.assertEqual(result.batches, 1)
            written = list(output_dir.glob("batch_*.parquet"))
            self.assertEqual(len(written), 1)

            out = pl.read_parquet(written[0])
            self.assertEqual(out["id2"].to_list(), [10, 20])

    def test_missing_dataset_raises(self) -> None:
        catalog = pli.LocalCatalog({"datasets": {"only": {"format": "parquet", "path": "data/raw"}}})
        with self.assertRaises(KeyError):
            catalog.resolve("missing")

    def test_invalid_catalog_payload(self) -> None:
        with self.assertRaises(ValueError):
            pli.LocalCatalog({"datasets": "not-a-mapping"})

    def test_missing_format_or_path(self) -> None:
        catalog = pli.LocalCatalog({"datasets": {"bad": {"path": "data/raw"}}})
        with self.assertRaises(ValueError):
            catalog.resolve("bad")
        catalog = pli.LocalCatalog({"datasets": {"bad": {"format": "parquet"}}})
        with self.assertRaises(ValueError):
            catalog.resolve("bad")

    def test_unsupported_catalog_file_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "catalog.yaml"
            path.write_text("datasets: {}")
            with self.assertRaises(ValueError):
                pli.LocalCatalog(path)

    def test_dataset_spec_delta_source(self) -> None:
        spec = pli.DatasetSpec(
            name="events",
            format="delta",
            path="/tmp/table",
            options={
                "start_offset": "latest",
                "starting_version": 3,
                "max_files_per_trigger": 10,
                "read_change_feed": True,
            },
        )
        source = spec.to_source()
        self.assertIsInstance(source, pli.DeltaSource)
        self.assertEqual(source.start_offset, "latest")
        self.assertEqual(source.starting_version, 3)

    def test_dataset_spec_auto_source(self) -> None:
        spec = pli.DatasetSpec(name="auto", format="auto", path="/tmp/data", options={"pattern": "*.csv"})
        source = spec.to_source()
        self.assertIsInstance(source, pli.AutoSource)

    def test_dataset_spec_files_source_max_bytes(self) -> None:
        spec = pli.DatasetSpec(
            name="events",
            format="parquet",
            path="/tmp/data",
            options={"max_bytes_per_trigger": 123},
        )
        source = spec.to_source()
        self.assertIsInstance(source, pli.FilesSource)
        self.assertEqual(source.max_bytes_per_trigger, 123)

    def test_dataset_spec_schema_evolution(self) -> None:
        spec = pli.DatasetSpec(
            name="events",
            format="parquet",
            path="/tmp/data",
            options={"schema_mode": "add_new_columns"},
        )
        schema_evolution = spec.to_schema_evolution()
        self.assertIsNotNone(schema_evolution)
        assert schema_evolution is not None
        self.assertEqual(schema_evolution.mode, "add_new_columns")

    def test_catalog_loads_json_and_toml_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "catalog.json"
            json_path.write_text('{"datasets": {"one": {"format": "parquet", "path": "/tmp"}}}')
            catalog = pli.LocalCatalog(json_path)
            self.assertEqual(catalog.resolve("one").format, "parquet")

            toml_path = Path(tmpdir) / "catalog.toml"
            toml_path.write_text('[datasets.one]\nformat="parquet"\npath="/tmp"\n')
            catalog = pli.LocalCatalog(toml_path)
            self.assertEqual(catalog.resolve("one").format, "parquet")
