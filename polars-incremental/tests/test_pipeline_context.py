import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from polars_incremental.errors import (
    MissingOptionError,
    PolarsIncrementalError,
    ReaderError,
)
from polars_incremental.pipeline import (
    Pipeline,
    _apply_schema_if_configured,
    _call_with_context,
    _pipeline_lock,
    _resolve_source,
    _run_loop,
    _stage_error,
)
from polars_incremental.schema import SchemaEvolution
from polars_incremental.sources.base import SourceSpec


class DummyBatch:
    def __init__(self, batch_id: int | None = None) -> None:
        self.batch_id = batch_id
        self.files = ["a.parquet"]


class TestPipelineContext(unittest.TestCase):
    def test_call_with_context_uses_positional_batch(self) -> None:
        batch = DummyBatch(batch_id=3)

        def func(data, batch):
            return (data, batch.batch_id)

        result = _call_with_context(
            func,
            first_arg="data",
            batch=batch,
            files=batch.files,
            include_files_kw=False,
            state=None,
        )
        self.assertEqual(result, ("data", 3))

    def test_call_with_context_signature_failure_falls_back(self) -> None:
        def func(data):
            return data

        with mock.patch("polars_incremental.pipeline.inspect.signature", side_effect=TypeError):
            result = _call_with_context(
                func,
                first_arg="ok",
                batch=DummyBatch(),
                files=[],
                include_files_kw=False,
                state=None,
            )
        self.assertEqual(result, "ok")

    def test_call_with_context_includes_files_kw(self) -> None:
        def func(data, files=None):
            return files

        batch = DummyBatch(batch_id=1)
        result = _call_with_context(
            func,
            first_arg="data",
            batch=batch,
            files=["a.parquet"],
            include_files_kw=True,
            state=None,
        )
        self.assertEqual(result, ["a.parquet"])

    def test_call_with_context_does_not_misbind_positional(self) -> None:
        def func(data, other):
            return other

        batch = DummyBatch(batch_id=6)
        with self.assertRaises(TypeError):
            _call_with_context(
                func,
                first_arg="data",
                batch=batch,
                files=batch.files,
                include_files_kw=False,
                state=None,
            )

    def test_call_with_context_does_not_misbind_batch_size(self) -> None:
        def func(data, batch_size):
            return batch_size

        batch = DummyBatch(batch_id=7)
        with self.assertRaises(TypeError):
            _call_with_context(
                func,
                first_arg="data",
                batch=batch,
                files=batch.files,
                include_files_kw=False,
                state=None,
            )

    def test_call_with_context_positional_batch_id(self) -> None:
        def func(data, _batch_id):
            return _batch_id

        batch = DummyBatch(batch_id=8)
        result = _call_with_context(
            func,
            first_arg="data",
            batch=batch,
            files=batch.files,
            include_files_kw=False,
            state=None,
        )
        self.assertEqual(result, 8)

    def test_call_with_context_supports_kwargs_only(self) -> None:
        def func(*, batch=None, files=None, state=None):
            return (batch.batch_id, files, state)

        batch = DummyBatch(batch_id=4)
        result = _call_with_context(
            func,
            first_arg="data",
            batch=batch,
            files=["a.parquet"],
            include_files_kw=True,
            state="state",
        )
        self.assertEqual(result, (4, ["a.parquet"], "state"))

    def test_call_with_context_passes_files_for_reader_kwargs_only(self) -> None:
        def func(*, files=None, batch=None):
            return (files, batch.batch_id)

        batch = DummyBatch(batch_id=5)
        result = _call_with_context(
            func,
            first_arg=["f.parquet"],
            batch=batch,
            files=["f.parquet"],
            include_files_kw=False,
            state=None,
        )
        self.assertEqual(result, (["f.parquet"], 5))

    def test_stage_error_without_batch_id(self) -> None:
        msg = _stage_error("reader", object())
        self.assertEqual(msg, "reader failed")

    def test_run_loop_idle_breaks_after_max_idle(self) -> None:
        class Source:
            def plan_batch(self):
                return None

            def commit_batch(self, batch, metadata=None):
                raise AssertionError("commit should not be called")

        with mock.patch("polars_incremental.pipeline.time.sleep") as sleep:
            result = _run_loop(
                source_impl=Source(),
                reader=lambda files: files,
                writer=lambda data: None,
                transform=None,
                schema_evolution=None,
                run_once=False,
                sleep=0.0,
                max_batches=None,
                sleep_when_idle=0.0,
                max_idle_loops=2,
                observer=None,
                state=None,
            )
        self.assertEqual(result.batches, 0)
        sleep.assert_called_once_with(0.0)

    def test_run_loop_idle_resets_after_success(self) -> None:
        class Source:
            def __init__(self) -> None:
                self.calls = 0
                self.sequence = [
                    DummyBatch(batch_id=1),
                    None,
                    DummyBatch(batch_id=2),
                    None,
                    None,
                ]

            def plan_batch(self):
                self.calls += 1
                if self.sequence:
                    return self.sequence.pop(0)
                return None

            def commit_batch(self, batch, metadata=None):
                return None

        source = Source()
        with mock.patch("polars_incremental.pipeline.time.sleep") as sleep:
            result = _run_loop(
                source_impl=source,
                reader=lambda files: files,
                writer=lambda data: None,
                transform=None,
                schema_evolution=None,
                run_once=False,
                sleep=0.25,
                max_batches=None,
                sleep_when_idle=0.0,
                max_idle_loops=2,
                observer=None,
                state=None,
            )
        self.assertEqual(result.batches, 2)
        self.assertEqual(source.calls, 5)
        idle_calls = [call for call in sleep.call_args_list if call == mock.call(0.0)]
        self.assertEqual(len(idle_calls), 2)

    def test_pipeline_missing_reader_raises(self) -> None:
        pipeline = Pipeline(
            source=SourceSpec(format="files", path="/tmp", options={}),
            checkpoint_dir="/tmp/checkpoint",
            reader=None,
            writer=lambda data: None,
        )
        with self.assertRaises(MissingOptionError):
            pipeline.run(once=True)

    def test_apply_schema_if_configured(self) -> None:
        schema_evolution = SchemaEvolution(mode="strict")
        data = {"rows": 1}
        result = _apply_schema_if_configured(data, source_impl=None, schema_evolution=schema_evolution)
        self.assertEqual(result, data)

    def test_resolve_source_rejects_invalid(self) -> None:
        with self.assertRaises(TypeError):
            _resolve_source(object(), "/tmp/checkpoint")

    def test_reader_error_wrapped(self) -> None:
        class Source:
            def plan_batch(self):
                return DummyBatch(batch_id=1)

            def commit_batch(self, batch, metadata=None):
                return None

        def reader(_files):
            raise ValueError("boom")

        with self.assertRaises(ReaderError):
            _run_loop(
                source_impl=Source(),
                reader=reader,
                writer=lambda data: None,
                transform=None,
                schema_evolution=None,
                run_once=True,
                sleep=0.0,
                max_batches=None,
                sleep_when_idle=None,
                max_idle_loops=None,
                observer=None,
                state=None,
            )

    def test_planning_error_passthrough(self) -> None:
        class Source:
            def plan_batch(self):
                raise PolarsIncrementalError("boom")

        with self.assertRaises(PolarsIncrementalError):
            _run_loop(
                source_impl=Source(),
                reader=lambda files: files,
                writer=lambda data: None,
                transform=None,
                schema_evolution=None,
                run_once=True,
                sleep=0.0,
                max_batches=None,
                sleep_when_idle=None,
                max_idle_loops=None,
                observer=None,
                state=None,
            )

    def test_reader_error_passthrough_for_incremental_error(self) -> None:
        class Source:
            def plan_batch(self):
                return DummyBatch(batch_id=1)

            def commit_batch(self, batch, metadata=None):
                return None

        def reader(_files):
            raise PolarsIncrementalError("boom")

        with self.assertRaises(PolarsIncrementalError):
            _run_loop(
                source_impl=Source(),
                reader=reader,
                writer=lambda data: None,
                transform=None,
                schema_evolution=None,
                run_once=True,
                sleep=0.0,
                max_batches=None,
                sleep_when_idle=None,
                max_idle_loops=None,
                observer=None,
                state=None,
            )

    def test_run_loop_max_batches_stops(self) -> None:
        class Source:
            def __init__(self) -> None:
                self.called = False

            def plan_batch(self):
                if self.called:
                    return None
                self.called = True
                return DummyBatch(batch_id=1)

            def commit_batch(self, batch, metadata=None):
                return None

        with mock.patch("polars_incremental.pipeline.time.sleep") as sleep:
            result = _run_loop(
                source_impl=Source(),
                reader=lambda files: files,
                writer=lambda data: None,
                transform=None,
                schema_evolution=None,
                run_once=False,
                sleep=0.0,
                max_batches=1,
                sleep_when_idle=None,
                max_idle_loops=None,
                observer=None,
                state=None,
            )
        self.assertEqual(result.batches, 1)
        sleep.assert_called_once_with(0.0)

    def test_pipeline_lock_uses_fcntl(self) -> None:
        fake_calls: list[int] = []

        class FakeFcntl:
            LOCK_EX = 1
            LOCK_UN = 2

            @staticmethod
            def flock(_handle, flag):
                fake_calls.append(flag)

        with mock.patch.object(
            __import__("polars_incremental.pipeline", fromlist=["fcntl"]),
            "fcntl",
            FakeFcntl,
        ):
            with mock.patch.dict(os.environ, {}, clear=True):
                with tempfile.TemporaryDirectory() as tmpdir:
                    with _pipeline_lock(Path(tmpdir)):
                        pass
        self.assertEqual(fake_calls, [FakeFcntl.LOCK_EX, FakeFcntl.LOCK_UN])

    def test_pipeline_lock_disabled_env_skips_lock(self) -> None:
        fake_calls: list[int] = []

        class FakeFcntl:
            LOCK_EX = 1
            LOCK_UN = 2

            @staticmethod
            def flock(_handle, flag):
                fake_calls.append(flag)

        with mock.patch.object(
            __import__("polars_incremental.pipeline", fromlist=["fcntl"]),
            "fcntl",
            FakeFcntl,
        ):
            with mock.patch.dict(os.environ, {"POLARS_INCREMENTAL_DISABLE_LOCK": "1"}):
                with tempfile.TemporaryDirectory() as tmpdir:
                    lock_path = Path(tmpdir) / ".pipeline.lock"
                    with _pipeline_lock(Path(tmpdir)):
                        pass
                    self.assertFalse(lock_path.exists())
        self.assertEqual(fake_calls, [])

    def test_pipeline_lock_fallback_creates_lock_file(self) -> None:
        pipeline_mod = __import__("polars_incremental.pipeline", fromlist=["fcntl"])
        with mock.patch.object(pipeline_mod, "fcntl", None):
            with mock.patch.dict(os.environ, {}, clear=True):
                with tempfile.TemporaryDirectory() as tmpdir:
                    lock_path = Path(tmpdir) / ".pipeline.lock"
                    self.assertFalse(lock_path.exists())
                    with _pipeline_lock(Path(tmpdir)):
                        self.assertTrue(lock_path.exists())
                    self.assertFalse(lock_path.exists())

    def test_pipeline_lock_fallback_raises_if_locked(self) -> None:
        pipeline_mod = __import__("polars_incremental.pipeline", fromlist=["fcntl"])
        with mock.patch.object(pipeline_mod, "fcntl", None):
            with mock.patch.dict(os.environ, {"POLARS_INCREMENTAL_LOCK_TIMEOUT": "0"}):
                with tempfile.TemporaryDirectory() as tmpdir:
                    lock_path = Path(tmpdir) / ".pipeline.lock"
                    lock_path.write_text("locked")
                    with self.assertRaises(RuntimeError):
                        with _pipeline_lock(Path(tmpdir)):
                            pass

    def test_pipeline_lock_fallback_clears_stale(self) -> None:
        pipeline_mod = __import__("polars_incremental.pipeline", fromlist=["fcntl"])
        with mock.patch.object(pipeline_mod, "fcntl", None):
            with tempfile.TemporaryDirectory() as tmpdir:
                lock_path = Path(tmpdir) / ".pipeline.lock"
                lock_path.write_text("pid=999999\nacquired_at=0\n")
                with mock.patch.dict(
                    os.environ,
                    {
                        "POLARS_INCREMENTAL_LOCK_TIMEOUT": "0",
                        "POLARS_INCREMENTAL_LOCK_STALE_SECONDS": "1",
                    },
                ):
                    with _pipeline_lock(Path(tmpdir)):
                        self.assertTrue(lock_path.exists())

    def test_pipeline_lock_fallback_clears_stale_pid(self) -> None:
        pipeline_mod = __import__("polars_incremental.pipeline", fromlist=["fcntl"])
        with mock.patch.object(pipeline_mod, "fcntl", None):
            with tempfile.TemporaryDirectory() as tmpdir:
                lock_path = Path(tmpdir) / ".pipeline.lock"
                lock_path.write_text("pid=12345\n")
                with mock.patch.object(pipeline_mod.os, "kill", side_effect=ProcessLookupError):
                    with mock.patch.dict(
                        os.environ,
                        {
                            "POLARS_INCREMENTAL_LOCK_TIMEOUT": "0",
                            "POLARS_INCREMENTAL_LOCK_STALE_SECONDS": "1",
                        },
                    ):
                        with _pipeline_lock(Path(tmpdir)):
                            self.assertTrue(lock_path.exists())

    def test_pipeline_lock_fallback_stale_unlink_failure_raises(self) -> None:
        pipeline_mod = __import__("polars_incremental.pipeline", fromlist=["fcntl"])
        with mock.patch.object(pipeline_mod, "fcntl", None):
            with tempfile.TemporaryDirectory() as tmpdir:
                lock_path = Path(tmpdir) / ".pipeline.lock"
                lock_path.write_text("pid=999999\nacquired_at=0\n")
                with mock.patch("pathlib.Path.unlink", side_effect=OSError("fail")):
                    with mock.patch.dict(
                        os.environ,
                        {
                            "POLARS_INCREMENTAL_LOCK_TIMEOUT": "0",
                            "POLARS_INCREMENTAL_LOCK_STALE_SECONDS": "1",
                        },
                    ):
                        with self.assertRaises(RuntimeError):
                            with _pipeline_lock(Path(tmpdir)):
                                pass

    def test_pipeline_lock_fallback_stale_uses_mtime(self) -> None:
        pipeline_mod = __import__("polars_incremental.pipeline", fromlist=["fcntl"])
        with mock.patch.object(pipeline_mod, "fcntl", None):
            with tempfile.TemporaryDirectory() as tmpdir:
                lock_path = Path(tmpdir) / ".pipeline.lock"
                lock_path.write_text("")
                os.utime(lock_path, (0, 0))
                with mock.patch.dict(
                    os.environ,
                    {
                        "POLARS_INCREMENTAL_LOCK_TIMEOUT": "0",
                        "POLARS_INCREMENTAL_LOCK_STALE_SECONDS": "1",
                    },
                ):
                    with _pipeline_lock(Path(tmpdir)):
                        self.assertTrue(lock_path.exists())
