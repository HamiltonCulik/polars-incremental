from __future__ import annotations

import inspect
import time
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .errors import (
    CommitError,
    MissingOptionError,
    PlanningError,
    PolarsIncrementalError,
    ReaderError,
    TransformError,
    WriterError,
)
from .observability import PipelineObserver
from .state import JobState
from .schema import SchemaEvolution
from .source import SourceConfig
from .sources.base import SourceSpec

try:
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover
    fcntl = None


@dataclass(frozen=True)
class RunResult:
    batches: int


@dataclass(frozen=True)
class Pipeline:
    source: SourceSpec | SourceConfig
    checkpoint_dir: str | Path
    reader: Callable[..., Any]
    writer: Callable[..., Any]
    transform: Callable[..., Any] | None = None
    schema_evolution: SchemaEvolution | None = None
    observer: PipelineObserver | None = None

    def run(
        self,
        *,
        once: bool = True,
        sleep: float = 1.0,
        max_batches: int | None = None,
        sleep_when_idle: float | None = None,
        max_idle_loops: int | None = None,
    ) -> RunResult:
        if self.reader is None or self.writer is None:
            raise MissingOptionError("reader and writer are required")

        source_impl = _resolve_source(self.source, self.checkpoint_dir)
        state = JobState(Path(self.checkpoint_dir) / "state")
        with _pipeline_lock(Path(self.checkpoint_dir)):
            return _run_loop(
                source_impl=source_impl,
                reader=self.reader,
                writer=self.writer,
                transform=self.transform,
                schema_evolution=self.schema_evolution,
                run_once=once,
                sleep=sleep,
                max_batches=max_batches,
                sleep_when_idle=sleep_when_idle,
                max_idle_loops=max_idle_loops,
                observer=self.observer,
                state=state,
            )


def _call_reader(
    reader: Callable[..., Any],
    files: list[str],
    batch: Any,
    state: JobState | None,
) -> Any:
    return _call_with_context(reader, files, batch, files, include_files_kw=False, state=state)


def _call_transform(
    transform: Callable[..., Any],
    data: Any,
    batch: Any,
    files: list[str],
    state: JobState | None,
) -> Any:
    return _call_with_context(transform, data, batch, files, include_files_kw=True, state=state)


def _call_writer(
    writer: Callable[..., Any],
    data: Any,
    batch: Any,
    files: list[str],
    state: JobState | None,
) -> Any:
    return _call_with_context(writer, data, batch, files, include_files_kw=True, state=state)


def _call_with_context(
    func: Callable[..., Any],
    first_arg: Any,
    batch: Any,
    files: list[str],
    *,
    include_files_kw: bool,
    state: JobState | None,
) -> Any:
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return func(first_arg)

    params = list(sig.parameters.values())
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    accepts_positional = any(
        p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for p in params
    ) or any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    kwargs: dict[str, Any] = {}

    if accepts_var_kw or "batch" in sig.parameters:
        kwargs["batch"] = batch
    if accepts_var_kw or "batch_id" in sig.parameters:
        kwargs["batch_id"] = getattr(batch, "batch_id", None)
    if include_files_kw and (accepts_var_kw or "files" in sig.parameters):
        kwargs["files"] = files
    if accepts_var_kw or "state" in sig.parameters:
        kwargs["state"] = state

    if accepts_positional:
        if kwargs:
            return func(first_arg, **kwargs)
        if len(params) >= 2:
            second = params[1]
            normalized_second = second.name.lstrip("_")
            if normalized_second == "batch":
                return func(first_arg, batch)
            if normalized_second == "batch_id":
                return func(first_arg, getattr(batch, "batch_id", None))
        return func(first_arg)

    context_keys = {"batch", "batch_id", "files", "state"}
    if not include_files_kw and "files" in sig.parameters and "files" not in kwargs:
        kwargs["files"] = first_arg
    else:
        non_context = [p for p in params if p.name not in context_keys]
        if len(non_context) == 1 and non_context[0].name not in kwargs:
            kwargs[non_context[0].name] = first_arg

    return func(**kwargs) if kwargs else func()


@contextmanager
def _pipeline_lock(checkpoint_dir: Path):
    if os.getenv("POLARS_INCREMENTAL_DISABLE_LOCK") == "1":
        yield
        return
    if fcntl is None:
        lock_path = checkpoint_dir / ".pipeline.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        timeout_raw = os.getenv("POLARS_INCREMENTAL_LOCK_TIMEOUT", "0")
        stale_raw = os.getenv("POLARS_INCREMENTAL_LOCK_STALE_SECONDS", "0")
        try:
            timeout_s = float(timeout_raw)
        except ValueError:
            timeout_s = 0.0
        try:
            stale_s = float(stale_raw)
        except ValueError:
            stale_s = 0.0
        start = time.monotonic()
        acquired = False
        while not acquired:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                stale_failed = False
                if stale_s > 0 and _is_stale_lock(lock_path, stale_s):
                    try:
                        lock_path.unlink(missing_ok=True)
                    except OSError:
                        stale_failed = True
                    else:
                        continue
                if stale_failed:
                    if timeout_s <= 0 or (time.monotonic() - start) >= timeout_s:
                        raise RuntimeError("pipeline lock already held")
                    time.sleep(min(0.1, max(0.01, timeout_s / 10)))
                    continue
                if timeout_s <= 0 or (time.monotonic() - start) >= timeout_s:
                    raise RuntimeError("pipeline lock already held")
                time.sleep(min(0.1, max(0.01, timeout_s / 10)))
                continue
            with os.fdopen(fd, "w") as handle:
                handle.write(f"pid={os.getpid()}\n")
                handle.write(f"acquired_at={time.time()}\n")
            acquired = True
        try:
            yield
        finally:
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass
        return
    lock_path = checkpoint_dir / ".pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _apply_schema_if_configured(
    data: Any, source_impl: Any, schema_evolution: SchemaEvolution | None
) -> Any:
    if schema_evolution is None:
        return data
    checkpoint = getattr(source_impl, "checkpoint", None)
    return schema_evolution.apply(data, checkpoint)


def _resolve_source(source: SourceSpec | SourceConfig, checkpoint_dir: str | Path) -> Any:
    spec = source.to_spec() if hasattr(source, "to_spec") else source
    if not isinstance(spec, SourceSpec):
        raise TypeError("source must be a SourceSpec or SourceConfig")
    return spec.with_checkpoint(checkpoint_dir)


def _is_stale_lock(lock_path: Path, stale_seconds: float) -> bool:
    if stale_seconds <= 0:
        return False
    pid: int | None = None
    acquired_at: float | None = None
    try:
        for line in lock_path.read_text().splitlines():
            if line.startswith("pid="):
                try:
                    pid = int(line.split("=", 1)[1].strip())
                except ValueError:
                    pid = None
            elif line.startswith("acquired_at="):
                try:
                    acquired_at = float(line.split("=", 1)[1].strip())
                except ValueError:
                    acquired_at = None
    except OSError:
        return False

    if acquired_at is None and pid is None:
        try:
            mtime = lock_path.stat().st_mtime
        except OSError:
            return False
        if (time.time() - mtime) >= stale_seconds:
            return True

    if acquired_at is not None:
        if (time.time() - acquired_at) >= stale_seconds:
            return True

    if pid is not None:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        except OSError:
            return False

    return False


def _run_loop(
    *,
    source_impl: Any,
    reader: Callable[..., Any],
    writer: Callable[..., Any],
    transform: Callable[..., Any] | None,
    schema_evolution: SchemaEvolution | None,
    run_once: bool,
    sleep: float,
    max_batches: int | None,
    sleep_when_idle: float | None,
    max_idle_loops: int | None,
    observer: PipelineObserver | None,
    state: JobState | None,
) -> RunResult:
    batches = 0
    idle_loops = 0

    def process_one() -> bool:
        nonlocal batches, idle_loops
        try:
            if observer is not None:
                observer.on_stage_start("plan", None)
            start = time.perf_counter()
            batch = source_impl.plan_batch()
            if observer is not None:
                observer.on_stage_end(
                    "plan",
                    getattr(batch, "batch_id", None),
                    time.perf_counter() - start,
                    metadata={"has_batch": batch is not None},
                )
        except PolarsIncrementalError as exc:
            if observer is not None:
                observer.on_error("plan", None, exc)
            raise
        except Exception as exc:
            if observer is not None:
                observer.on_error("plan", None, exc)
            raise PlanningError("Failed to plan batch") from exc
        if batch is None:
            return False
        files = list(getattr(batch, "files", []) or [])
        if observer is not None:
            observer.on_batch_planned(batch, files)
        try:
            if observer is not None:
                observer.on_stage_start("reader", getattr(batch, "batch_id", None))
            start = time.perf_counter()
            data = _call_reader(reader, files, batch, state)
            if observer is not None:
                observer.on_stage_end(
                    "reader",
                    getattr(batch, "batch_id", None),
                    time.perf_counter() - start,
                    metadata={"file_count": len(files)},
                )
        except PolarsIncrementalError as exc:
            if observer is not None:
                observer.on_error("reader", getattr(batch, "batch_id", None), exc)
            raise
        except Exception as exc:
            if observer is not None:
                observer.on_error("reader", getattr(batch, "batch_id", None), exc)
            raise ReaderError(_stage_error("reader", batch)) from exc
        data = _apply_schema_if_configured(data, source_impl, schema_evolution)
        if transform is not None:
            try:
                if observer is not None:
                    observer.on_stage_start("transform", getattr(batch, "batch_id", None))
                start = time.perf_counter()
                data = _call_transform(transform, data, batch, files, state)
                if observer is not None:
                    observer.on_stage_end(
                        "transform",
                        getattr(batch, "batch_id", None),
                        time.perf_counter() - start,
                    )
            except PolarsIncrementalError as exc:
                if observer is not None:
                    observer.on_error("transform", getattr(batch, "batch_id", None), exc)
                raise
            except Exception as exc:
                if observer is not None:
                    observer.on_error("transform", getattr(batch, "batch_id", None), exc)
                raise TransformError(_stage_error("transform", batch)) from exc
        try:
            if observer is not None:
                observer.on_stage_start("writer", getattr(batch, "batch_id", None))
            start = time.perf_counter()
            result = _call_writer(writer, data, batch, files, state)
            if observer is not None:
                observer.on_stage_end(
                    "writer",
                    getattr(batch, "batch_id", None),
                    time.perf_counter() - start,
                    metadata=result if isinstance(result, dict) else None,
                )
        except PolarsIncrementalError as exc:
            if observer is not None:
                observer.on_error("writer", getattr(batch, "batch_id", None), exc)
            raise
        except Exception as exc:
            if observer is not None:
                observer.on_error("writer", getattr(batch, "batch_id", None), exc)
            raise WriterError(_stage_error("writer", batch)) from exc
        metadata = result if isinstance(result, dict) else {}
        try:
            if observer is not None:
                observer.on_stage_start("commit", getattr(batch, "batch_id", None))
            start = time.perf_counter()
            source_impl.commit_batch(batch, metadata=metadata)
            if observer is not None:
                observer.on_stage_end(
                    "commit",
                    getattr(batch, "batch_id", None),
                    time.perf_counter() - start,
                    metadata=metadata or None,
                )
                observer.on_batch_committed(getattr(batch, "batch_id", None), metadata=metadata or None)
        except PolarsIncrementalError as exc:
            if observer is not None:
                observer.on_error("commit", getattr(batch, "batch_id", None), exc)
            raise
        except Exception as exc:
            if observer is not None:
                observer.on_error("commit", getattr(batch, "batch_id", None), exc)
            raise CommitError(_stage_error("commit", batch)) from exc
        batches += 1
        idle_loops = 0
        return True

    if run_once:
        process_one()
        return RunResult(batches=batches)

    while True:
        if max_batches is not None and batches >= max_batches:
            break
        if not process_one():
            if sleep_when_idle is None:
                break
            idle_loops += 1
            if max_idle_loops is not None and idle_loops >= max_idle_loops:
                break
            time.sleep(sleep_when_idle)
            continue
        time.sleep(sleep)
    return RunResult(batches=batches)


def _stage_error(stage: str, batch: Any) -> str:
    batch_id = getattr(batch, "batch_id", None)
    if batch_id is None:
        return f"{stage} failed"
    return f"{stage} failed for batch_id={batch_id}"
