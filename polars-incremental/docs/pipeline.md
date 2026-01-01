# Pipeline API

## `pli.Pipeline`

```python
pipeline = pli.Pipeline(
    source=pli.FilesSource(path="data/raw", file_format="parquet"),
    checkpoint_dir="data/checkpoints/raw",
    reader=reader,
    writer=writer,
    transform=transform,
    schema_evolution=pli.SchemaEvolution(mode="add_new_columns"),
    observer=pli.LoggingObserver(),
)

pipeline.run(once=True, sleep=1.0, max_batches=None)
```

### Parameters

- `source`: `FilesSource`, `DeltaSource`, or `SourceSpec`.
- `checkpoint_dir`: where offset/commit logs live.
- `reader`: function that takes `files` and returns a `DataFrame` or `LazyFrame`.
- `writer`: function called last; commits only after this succeeds.
- `transform`: optional function applied between reader and writer.
- `schema_evolution`: optional `SchemaEvolution` config.
- `observer`: optional `PipelineObserver` for logging/metrics hooks.

### `Pipeline.run(...)`

- `once`: process at most one batch (default `True`).
- `sleep`: delay between batches when looping.
- `max_batches`: stop after N batches when looping.
- `sleep_when_idle`: if set, sleep this many seconds when no batch is available and continue polling.
- `max_idle_loops`: stop after N consecutive idle polls (only used when `sleep_when_idle` is set).

## Reader / transform / writer

- `reader(files)` returns a `DataFrame` or `LazyFrame`.
- `transform(data)` is optional; it receives the output of `reader`.
- `writer(data)` is called last; the checkpoint commit happens **only after writer succeeds**.

### Context kwargs (exact behavior)

The pipeline can pass three context kwargs:

- `batch`: the source-specific batch object (always has `batch_id`).
- `batch_id`: the numeric batch id.
- `files`: the file list selected for this batch.
- `state`: per-pipeline `JobState` helper for cross-batch state.

The pipeline only passes these when your function signature includes them (or you accept `**kwargs`):

```python
def reader(files, batch=None, batch_id=None, state=None):
    ...

def transform(df_or_lf, batch=None, batch_id=None, files=None, state=None):
    ...

def writer(df_or_lf, batch=None, batch_id=None, files=None, state=None):
    ...
```

Notes:
- `files` is the list of inputs chosen for that batch (file paths or Delta data files).
- `batch` is a source-specific batch object that at least carries `batch_id`.
- `reader` always receives `files` as its first positional argument; the `files` kwarg is only passed to `transform` and `writer`.

## Return values

- `Pipeline.run(...)` returns `RunResult(batches=...)`.
- If your writer returns a `dict`, it is stored as commit metadata.

Commit metadata is written under `checkpoint_dir/commits/<batch_id>.json`.

## Loop behavior

- `once=True` processes at most one batch and returns.
- `once=False` loops, sleeping `sleep` seconds between batches.
- `max_batches` limits total batches in a looping run.
- If no new batch is available, the reader/writer are not called and `batches` remains 0.

## Execution engine

polars-incremental is engine-agnostic. If your reader returns a `LazyFrame`, you control how it executes (for example, you can call `collect(engine="streaming")` or use `sink_parquet`/`sink_csv` in your writer). The pipeline never calls `collect()` for you. Schema evolution works for both DataFrame and LazyFrame inputs.

## Example with source options

```python
pipeline = pli.Pipeline(
    source=pli.FilesSource(
        path="data/raw",
        file_format="parquet",
        pattern="events_*.parquet",
        max_files_per_trigger=100,
    ),
    checkpoint_dir="data/checkpoints/raw",
    reader=reader,
    writer=writer,
)

pipeline.run(once=True)
```
