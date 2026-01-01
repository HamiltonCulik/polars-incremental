# polars-incremental docs

Start here if you're using the library. This folder focuses on usage and principles of polars-incremental.

## What this library is

- A Polars-first, Spark-style checkpointed ingestion loop.
- Incremental reads for files and Delta tables.
- Schema evolution helpers and maintenance utilities.
- Local-first and scheduler-agnostic (use cron/systemd/orchestrators externally).
- Optional Polars-only CDC merge helper via `apply_cdc` (manage storage yourself).

## Core idea

All ingestion runs go through one path (preferred):

```python
pli.Pipeline(source, checkpoint_dir, reader, writer, transform=None, schema_evolution=None).run(...)
```

The checkpoint decides which inputs to process; your Polars code stays Polars-native.

## Quick links

- Getting started: `getting-started.md`
- Concepts: `concepts.md`
- Pipeline API: `pipeline.md`
- Sources and options: `sources.md`
- Checkpoint internals: `checkpoints.md`
- Schema evolution: `schema-evolution.md`
- Testing pipelines: `testing.md`
- Performance guidance: `performance.md`
- Errors and failure semantics: `errors.md`
- Observability and logging: `observability.md`
- Migration and checkpoint changes: `migrations.md`
- Job state: `state.md`
- Delta + CDC/CDF: `delta.md`
- Maintenance helpers: `maintenance.md`
- Catalog: `catalog.md`
- Examples index: `examples.md`
- Advanced patterns: `advanced-patterns.md`
- FAQ: `faq.md`
- API refactor proposal: `api-refactor-proposal.md`

## Minimal example

```python
import polars as pl
import polars_incremental as pli

def reader(files):
    return pl.scan_parquet(files)

def writer(lf, batch_id=None):
    lf.sink_parquet(f"data/out/batch_{batch_id}.parquet")

pipeline = pli.Pipeline(
    source=pli.FilesSource(path="data/raw", file_format="parquet"),
    checkpoint_dir="data/checkpoints/out",
    reader=reader,
    writer=writer,
)

pipeline.run(once=True)
```
