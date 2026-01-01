import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

import polars_incremental as pli
from polars_incremental.sources import SourceSpec
from polars_incremental.sources.delta import DeltaSource
from polars_incremental.sources.file import FileSource


class TestPipeline(unittest.TestCase):
    def test_observer_hooks(self) -> None:
        class RecordingObserver:
            def __init__(self) -> None:
                self.events: list[tuple[str, int | None]] = []

            def on_batch_planned(self, batch, files) -> None:
                self.events.append(("batch_planned", batch.batch_id))

            def on_stage_start(self, stage: str, batch_id: int | None) -> None:
                self.events.append((f"{stage}_start", batch_id))

            def on_stage_end(self, stage: str, batch_id: int | None, duration_s: float, metadata=None) -> None:
                self.events.append((f"{stage}_end", batch_id))

            def on_batch_committed(self, batch_id: int | None, metadata=None) -> None:
                self.events.append(("batch_committed", batch_id))

            def on_error(self, stage: str, batch_id: int | None, exc: Exception) -> None:
                self.events.append((f"{stage}_error", batch_id))

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir.mkdir(parents=True, exist_ok=True)

            pl.DataFrame({"id": [1]}).write_parquet(input_dir / "part-0000.parquet")

            def reader(files):
                return pl.scan_parquet(files)

            def writer(lf, batch_id=None):
                return {"rows": lf.collect().height}

            observer = RecordingObserver()

            pipeline = pli.Pipeline(
                source=pli.FilesSource(path=input_dir, file_format="parquet"),
                checkpoint_dir=checkpoint_dir,
                reader=reader,
                writer=writer,
                observer=observer,
            )
            pipeline.run(once=True)

            expected_prefix = [
                ("plan_start", None),
                ("plan_end", 0),
                ("batch_planned", 0),
                ("reader_start", 0),
                ("reader_end", 0),
                ("writer_start", 0),
                ("writer_end", 0),
                ("commit_start", 0),
                ("commit_end", 0),
                ("batch_committed", 0),
            ]
            self.assertEqual(observer.events[: len(expected_prefix)], expected_prefix)

    def test_run_with_reader_transform_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            sink_dir = Path(tmpdir) / "sink"
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir.mkdir(parents=True, exist_ok=True)
            sink_dir.mkdir(parents=True, exist_ok=True)

            pl.DataFrame({"id": [1, 2], "value": [10, 20]}).write_parquet(
                input_dir / "part-0000.parquet"
            )

            def reader(files):
                return pl.scan_parquet(files)

            def transform(lf):
                return lf.with_columns((pl.col("value") * 2).alias("value2")).filter(
                    pl.col("id") == 2
                )

            def writer(lf, batch):
                df = lf.collect()
                out_path = sink_dir / f"batch_{batch.batch_id}.parquet"
                df.write_parquet(out_path)
                return {"path": str(out_path), "rows": df.height}

            pipeline = pli.Pipeline(
                source=pli.FilesSource(path=input_dir, file_format="parquet"),
                checkpoint_dir=checkpoint_dir,
                reader=reader,
                transform=transform,
                writer=writer,
            )
            result = pipeline.run(once=True)

            self.assertEqual(result.batches, 1)
            written = list(sink_dir.glob("batch_*.parquet"))
            self.assertEqual(len(written), 1)

            out = pl.read_parquet(written[0])
            self.assertEqual(out.shape, (1, 3))
            self.assertEqual(out["value2"].to_list(), [40])

    def test_state_injection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir.mkdir(parents=True, exist_ok=True)

            pl.DataFrame({"id": [1]}).write_parquet(input_dir / "part-0000.parquet")

            def reader(files, state=None):
                assert state is not None
                state.save_json("seen", {"count": 1})
                return pl.read_parquet(files)

            def writer(df, state=None, batch_id=None):
                assert state is not None
                payload = state.load_json("seen", default={})
                return {"seen": payload.get("count"), "batch_id": batch_id}

            pipeline = pli.Pipeline(
                source=pli.FilesSource(path=input_dir, file_format="parquet"),
                checkpoint_dir=checkpoint_dir,
                reader=reader,
                writer=writer,
            )
            pipeline.run(once=True)

            state_dir = checkpoint_dir / "state"
            self.assertTrue((state_dir / "seen.json").exists())

    def test_checkpoint_not_committed_on_sink_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            sink_dir = Path(tmpdir) / "sink"
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir.mkdir(parents=True, exist_ok=True)
            sink_dir.mkdir(parents=True, exist_ok=True)

            pl.DataFrame({"id": [1]}).write_parquet(input_dir / "part-0000.parquet")

            def reader(files):
                return pl.scan_parquet(files)

            def writer(_lf, _batch):
                raise RuntimeError("sink failed")

            pipeline = pli.Pipeline(
                source=pli.FilesSource(path=input_dir, file_format="parquet"),
                checkpoint_dir=checkpoint_dir,
                reader=reader,
                writer=writer,
            )
            with self.assertRaises(pli.WriterError):
                pipeline.run(once=True)

            commit_dir = checkpoint_dir / "commits"
            commits = list(commit_dir.glob("*.json"))
            self.assertEqual(len(commits), 0)

            offset_dir = checkpoint_dir / "offsets"
            offsets = list(offset_dir.glob("*.json"))
            self.assertEqual(len(offsets), 1)

    def test_retry_reuses_offset_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            sink_dir = Path(tmpdir) / "sink"
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir.mkdir(parents=True, exist_ok=True)
            sink_dir.mkdir(parents=True, exist_ok=True)

            pl.DataFrame({"id": [1]}).write_parquet(input_dir / "part-0000.parquet")

            def reader(files):
                return pl.scan_parquet(files)

            def writer_factory():
                calls = {"count": 0}

                def writer(lf, batch):
                    calls["count"] += 1
                    if calls["count"] == 1:
                        raise RuntimeError("sink failed")
                    df = lf.collect()
                    out_path = sink_dir / f"batch_{batch.batch_id}.parquet"
                    df.write_parquet(out_path)
                    return {"path": str(out_path)}

                return writer

            writer = writer_factory()

            pipeline = pli.Pipeline(
                source=pli.FilesSource(path=input_dir, file_format="parquet"),
                checkpoint_dir=checkpoint_dir,
                reader=reader,
                writer=writer,
            )
            with self.assertRaises(pli.WriterError):
                pipeline.run(once=True)

            pipeline.run(once=True)

            offset_dir = checkpoint_dir / "offsets"
            commit_dir = checkpoint_dir / "commits"
            self.assertEqual(len(list(offset_dir.glob("*.json"))), 1)
            commits = list(commit_dir.glob("*.json"))
            self.assertEqual(len(commits), 1)

            payload = json.loads((commit_dir / "0.json").read_text())
            self.assertEqual(payload.get("batch_id"), 0)

    def test_pipeline_does_not_wrap_polars_incremental_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir.mkdir(parents=True, exist_ok=True)

            pl.DataFrame({"id": [1]}).write_parquet(input_dir / "part-0000.parquet")

            def reader(files):
                return pl.scan_parquet(files)

            def writer(_lf, _batch):
                raise pli.PolarsIncrementalError("boom")

            with self.assertRaises(pli.PolarsIncrementalError):
                pipeline = pli.Pipeline(
                    source=pli.FilesSource(path=input_dir, file_format="parquet"),
                    checkpoint_dir=checkpoint_dir,
                    reader=reader,
                    writer=writer,
                )
                pipeline.run(once=True)

    def test_reader_can_accept_batch_kw(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir.mkdir(parents=True, exist_ok=True)

            pl.DataFrame({"id": [1]}).write_parquet(input_dir / "part-0000.parquet")

            def reader(files, batch=None):
                self.assertIsNotNone(batch)
                return pl.scan_parquet(files)

            def writer(_lf, _batch):
                return {"ok": True}

            pipeline = pli.Pipeline(
                source=pli.FilesSource(path=input_dir, file_format="parquet"),
                checkpoint_dir=checkpoint_dir,
                reader=reader,
                writer=writer,
            )
            pipeline.run(once=True)

    def test_transform_can_accept_batch_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            sink_dir = Path(tmpdir) / "sink"
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir.mkdir(parents=True, exist_ok=True)
            sink_dir.mkdir(parents=True, exist_ok=True)

            pl.DataFrame({"id": [1]}).write_parquet(input_dir / "part-0000.parquet")

            def reader(files):
                return pl.scan_parquet(files)

            def transform(lf, batch_id=None):
                self.assertEqual(batch_id, 0)
                return lf

            def writer(lf, batch_id=None):
                df = lf.collect()
                out_path = sink_dir / f"batch_{batch_id}.parquet"
                df.write_parquet(out_path)
                return {"path": str(out_path)}

            pipeline = pli.Pipeline(
                source=pli.FilesSource(path=input_dir, file_format="parquet"),
                checkpoint_dir=checkpoint_dir,
                reader=reader,
                transform=transform,
                writer=writer,
            )
            pipeline.run(once=True)

    def test_transform_error_wrapped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir.mkdir(parents=True, exist_ok=True)

            pl.DataFrame({"id": [1]}).write_parquet(input_dir / "part-0000.parquet")

            def reader(files):
                return pl.read_parquet(files)

            def transform(_df):
                raise ValueError("boom")

            def writer(df):
                return {"rows": df.height}

            pipeline = pli.Pipeline(
                source=pli.FilesSource(path=input_dir, file_format="parquet"),
                checkpoint_dir=checkpoint_dir,
                reader=reader,
                transform=transform,
                writer=writer,
            )
            with self.assertRaises(pli.TransformError):
                pipeline.run(once=True)

    def test_commit_error_wrapped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir.mkdir(parents=True, exist_ok=True)

            pl.DataFrame({"id": [1]}).write_parquet(input_dir / "part-0000.parquet")

            def reader(files):
                return pl.read_parquet(files)

            def writer(df):
                return {"rows": df.height}

            pipeline = pli.Pipeline(
                source=pli.FilesSource(path=input_dir, file_format="parquet"),
                checkpoint_dir=checkpoint_dir,
                reader=reader,
                writer=writer,
            )

            with patch("polars_incremental.sources.file.FileSource.commit_batch", side_effect=RuntimeError("boom")):
                with self.assertRaises(pli.CommitError):
                    pipeline.run(once=True)

    def test_planning_error_wrapped(self) -> None:
        class FakeSource:
            def plan_batch(self):
                raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            input_dir.mkdir(parents=True, exist_ok=True)

            pl.DataFrame({"id": [1]}).write_parquet(input_dir / "part-0000.parquet")
            spec = SourceSpec(format="parquet", path=input_dir, options={})

            def reader(files):
                return pl.read_parquet(files)

            def writer(df):
                return {"rows": df.height}

            pipeline = pli.Pipeline(
                source=spec,
                checkpoint_dir=checkpoint_dir,
                reader=reader,
                writer=writer,
            )

            with patch("polars_incremental.sources.base.SourceSpec.with_checkpoint", return_value=FakeSource()):
                with self.assertRaises(pli.PlanningError):
                    pipeline.run(once=True)

    def test_source_auto_prefers_file_format_option(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            spec = SourceSpec(format="auto", path=data_dir, options={"file_format": "csv"})
            source = spec.with_checkpoint(Path(tmpdir) / "checkpoint")

            self.assertIsInstance(source, FileSource)
            self.assertEqual(source.file_format, "csv")
            self.assertEqual(source.pattern, "*.csv")

    def test_source_auto_uses_pattern_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            spec = SourceSpec(format="auto", path=data_dir, options={"pattern": "*.csv"})
            source = spec.with_checkpoint(Path(tmpdir) / "checkpoint")

            self.assertIsInstance(source, FileSource)
            self.assertEqual(source.file_format, "csv")
            self.assertEqual(source.pattern, "*.csv")

    def test_source_auto_detects_delta_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "delta_table"
            (data_dir / "_delta_log").mkdir(parents=True, exist_ok=True)

            spec = SourceSpec(format="auto", path=data_dir, options={})
            source = spec.with_checkpoint(Path(tmpdir) / "checkpoint")

            self.assertIsInstance(source, DeltaSource)

    def test_source_auto_defaults_to_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            spec = SourceSpec(format="auto", path=data_dir, options={})
            source = spec.with_checkpoint(Path(tmpdir) / "checkpoint")

            self.assertIsInstance(source, FileSource)
            self.assertEqual(source.file_format, "parquet")
            self.assertEqual(source.pattern, "*.parquet")

    def test_source_auto_uses_excel_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            spec = SourceSpec(format="auto", path=data_dir, options={"pattern": "*.xlsx"})
            source = spec.with_checkpoint(Path(tmpdir) / "checkpoint")

            self.assertIsInstance(source, FileSource)
            self.assertEqual(source.file_format, "excel")
            self.assertEqual(source.pattern, "*.xlsx")

    def test_source_explicit_excel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            spec = SourceSpec(format="excel", path=data_dir, options={})
            source = spec.with_checkpoint(Path(tmpdir) / "checkpoint")

            self.assertIsInstance(source, FileSource)
            self.assertEqual(source.file_format, "excel")
            self.assertEqual(source.pattern, "*.xlsx")

    def test_source_auto_uses_ndjson_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            spec = SourceSpec(format="auto", path=data_dir, options={"pattern": "*.jsonl"})
            source = spec.with_checkpoint(Path(tmpdir) / "checkpoint")

            self.assertIsInstance(source, FileSource)
            self.assertEqual(source.file_format, "ndjson")
            self.assertEqual(source.pattern, "*.jsonl")

    def test_source_explicit_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            spec = SourceSpec(format="json", path=data_dir, options={})
            source = spec.with_checkpoint(Path(tmpdir) / "checkpoint")

            self.assertIsInstance(source, FileSource)
            self.assertEqual(source.file_format, "json")
            self.assertEqual(source.pattern, "*.json")

    def test_source_explicit_ndjson(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            spec = SourceSpec(format="ndjson", path=data_dir, options={})
            source = spec.with_checkpoint(Path(tmpdir) / "checkpoint")

            self.assertIsInstance(source, FileSource)
            self.assertEqual(source.file_format, "ndjson")
            self.assertEqual(source.pattern, "*.ndjson")

    def test_source_explicit_avro(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            spec = SourceSpec(format="avro", path=data_dir, options={})
            source = spec.with_checkpoint(Path(tmpdir) / "checkpoint")

            self.assertIsInstance(source, FileSource)
            self.assertEqual(source.file_format, "avro")
            self.assertEqual(source.pattern, "*.avro")
