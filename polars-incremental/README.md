# polars-incremental

Polars-native incremental ingestion helpers, with checkpointing, schema evolution,
CDC helpers, and stateful processing. Includes a Polars-only CDC merge helper
(`apply_cdc`) and Delta CDC helper (`apply_cdc_delta`).

## What this does

- Spark Structured Streaming-style **offset logs** + **commit logs**.
- Incremental reads for **files** and **Delta tables** (versioned).
- Incremental ingestion is the core; CDC is just one supported use case.
- Delta sink with **mergeSchema** support.
- Schema evolution policies (strict, add columns, coerce, type_widen) + optional rescue.
- CDC apply helper for Delta sinks (merge/append_only).
- Start offsets and file overwrite tracking (optional).
- Delta Change Data Feed (CDF) read support (optional).
- Per-pipeline state directory for user-managed state (JobState).
- Optional pipeline observability hooks (logging or custom observers).

## Quick start

From this folder:

```
uv pip install -e .
```

Polars-native flow (reader -> transform -> writer with checkpoints):

```python
import polars as pl
import polars_incremental as pli

def reader(files):
    return pl.scan_parquet(files)

def transform(lf):
    return (
        lf.with_columns(pl.col("ts").cast(pl.Datetime("ms")))
          .group_by_dynamic("ts", every="1m")
          .agg(pl.len().alias("n"))
    )

def writer(lf, batch_id=None):
    df = lf.collect()
    df.write_delta("data/delta/events_1m")

source = pli.FilesSource(
    path="data/raw",
    file_format="parquet",
)

pipeline = pli.Pipeline(
    source=source,
    checkpoint_dir="data/checkpoints/events_1m",
    reader=reader,
    transform=transform,
    writer=writer,
)

pipeline.run(once=True)
```

`reader(files)` receives the batch file list chosen by the checkpoint, so your Polars code can stay the same.

`Pipeline` + `FilesSource`/`DeltaSource` objects are the preferred API.

You can still set `source` explicitly to `"files"`, `"parquet"`, `"csv"`, `"excel"`, `"json"`, `"ndjson"`, `"avro"`, or `"delta"`.

Schema evolution can be configured explicitly:

```python
schema = pli.SchemaEvolution(mode="add_new_columns", rescue_mode="column")
pipeline = pli.Pipeline(
    source=source,
    checkpoint_dir="data/checkpoints/events_1m",
    reader=reader,
    transform=transform,
    writer=writer,
    schema_evolution=schema,
)
```

Catalog-driven (optional):

```python
import polars as pl
import polars_incremental as pli

catalog = pli.LocalCatalog({
    "datasets": {
        "raw_events": {"format": "parquet", "path": "data/raw"},
        "events_out": {
            "format": "parquet",
            "path": "data/out/events",
        },
    }
})

source = catalog.get_source("raw_events")
sink = catalog.resolve("events_out")

def reader(files):
    return pl.scan_parquet(files).filter(pl.col("id") > 0)

def writer(lf, batch_id=None):
    lf.sink_parquet(f"{sink.path}/batch_{batch_id}.parquet")

pipeline = pli.Pipeline(
    source=source,
    checkpoint_dir="data/checkpoints/events_out",
    reader=reader,
    writer=writer,
)

pipeline.run(once=True)
```

## State (JobState)

Each checkpoint has a sibling `state/` directory that you can use for small, job-owned state.
The library passes `state` into reader/transform/writer if you accept it:

```python
def transform(df, state=None):
    watermark = state.load_json("watermark", default={"value": None})
    # update watermark and persist it
    state.save_json("watermark", {"value": "2024-01-01T00:00:00Z"})
    return df
```

Writes are atomic (temp file + replace). See `docs/state.md` and advanced examples.

## Observability

Attach a `PipelineObserver` to get structured stage/batch events:

```python
observer = pli.LoggingObserver()
pipeline = pli.Pipeline(..., observer=observer)
```

## Backpressure / batching

Control batch size with:

- `max_files_per_trigger`
- `max_bytes_per_trigger`

Control idle behavior in the run loop with:

- `sleep_when_idle`
- `max_idle_loops`

## Layout

- `src/polars_incremental/` — library code (public API + sources/sinks/checkpoints).
- `examples/` — runnable usage examples (no CLI flags).
- `tests/` — `unittest` coverage (not part of the package).
- `docs/dev-notes.md` — internal notes and roadmap ideas.

Repo root (outside this package) also includes `scripts/` for ad-hoc testing.

## Checkpoint format (Spark-style)

```
checkpoint_dir/
  metadata.json
  offsets/
    0.json
    1.json
  commits/
    0.json
    1.json
```

- **Offsets** store the planned inputs for each batch.
- **Commits** mark a batch as successfully processed.
- If an offset exists without a commit, the batch is retried on restart (at-least-once).

## Maintenance helpers

```python
import polars_incremental as pli

pli.cleanup_checkpoint("data/checkpoints/events_out", keep_last_n=5)
pli.vacuum_delta_table("data/delta/events", retention_hours=168.0, dry_run=True)
pli.optimize_delta_table("data/delta/events", mode="compact")
```

## Notes

- Delta writes do not fall back; failures propagate.
- Use one writer per checkpoint location to avoid concurrent processing conflicts.
- The checkpoint loop controls which files are passed to `reader(files)` each batch.
- Schema evolution applies to `DataFrame` and `LazyFrame` inputs (applies when enabled).
- Checkpoint paths are job config, not catalog metadata.
- Start offsets are persisted in checkpoint metadata; if you need to change them, use a new checkpoint dir or `reset_checkpoint_start_offset(...)`.
- Errors and failure semantics are documented in `docs/errors.md`.
- Testing guidance is in `docs/testing.md`.
- Performance guidance is in `docs/performance.md`.
- Migration guidance is in `docs/migrations.md`.
- Advanced patterns are in `docs/advanced-patterns.md` and `examples/advanced-patterns/`.
