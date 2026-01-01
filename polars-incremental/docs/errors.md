# Errors and failure semantics

polars-incremental raises explicit error types so you can handle failures deterministically. All custom errors inherit from `PolarsIncrementalError`.

## Pipeline errors

These wrap exceptions thrown while running `Pipeline.run(...)`. The original exception is available via `__cause__`.

- `PipelineError`: base class for pipeline execution failures.
- `PlanningError`: raised when a source fails to plan a batch.
- `ReaderError`: raised when the reader callable fails.
- `TransformError`: raised when the transform callable fails.
- `WriterError`: raised when the writer callable fails.
- `CommitError`: raised when committing a batch fails.

## Configuration and schema errors

- `MissingOptionError`: required configuration is missing.
- `UnsupportedFormatError`: unsupported source format requested.
- `SchemaEvolutionError`: schema mode/rescue rules are violated.

## Delta/CDF and sink errors

- `ChangeDataFeedError`: requested CDF read, but CDF is unavailable.

## What to catch

```python
import polars_incremental as pli

pipeline = pli.Pipeline(...)

try:
    pipeline.run(...)
except pli.ReaderError as exc:
    # reader-specific issue; inspect exc.__cause__ for the original error
    raise
except pli.WriterError as exc:
    # writer-specific issue; may need idempotent handling
    raise
except pli.SchemaEvolutionError:
    # schema mismatch or invalid schema mode
    raise
except pli.PolarsIncrementalError:
    # any other polars-incremental error
    raise
```

## Retryability and checkpoint state

polars-incremental is **at-least-once**. Offsets are written before processing a batch, and commits are written **only after** the writer succeeds.

- If planning fails before an offset is written, no new checkpoint state is created.
- If the reader/transform/writer fails **after** an offset is written, there is **no commit**. On restart, the same batch is retried with the same `batch_id` and inputs.
- If the writer succeeds but commit fails, outputs may already exist. Your writer should be idempotent or use `batch_id` in output paths to avoid duplicates.

In general:
- **Reader/Transform/Writer errors** are often retryable once the underlying issue is fixed.
- **Schema/config errors** are usually non-retryable until configuration or data changes.
- **I/O errors** (checkpoint or Delta) may be transient; retries are usually safe.

## Quick guide: retryable vs non-retryable

This is a practical guide (not a hard rule):

- Retryable after fix: `ReaderError`, `TransformError`, `WriterError`, `CommitError`, `PlanningError`.
- Non-retryable until config/data changes: `SchemaEvolutionError`, `MissingOptionError`, `UnsupportedFormatError`.

The underlying exception (`exc.__cause__`) tells you whether the issue is transient (I/O) or deterministic (schema/config).

## Recovery patterns

### 1) Fix and retry (default)

If a batch fails after the offset is written, rerunning the pipeline will retry the **same** batch id and input set. This is the safest option for at-least-once semantics.

### 2) Idempotent writer

If your writer is idempotent (for example, using `batch_id` in output paths or overwriting output for that batch), retries are safe.

### 3) Reprocess from scratch

If you want to reprocess everything, use a **new checkpoint directory**. This avoids mixing old offsets with new logic.

## Observability

For logging and metrics hooks, see `docs/observability.md`.
