# Testing pipelines

This page shows practical patterns for testing pipelines that use checkpoints.

## Use a temporary checkpoint per test

Checkpoint state is persisted across runs, so tests should use a fresh directory to
avoid cross-test contamination.

```python
import tempfile
from pathlib import Path
import polars as pl
import polars_incremental as pli

with tempfile.TemporaryDirectory() as tmpdir:
    base = Path(tmpdir)
    raw_dir = base / "raw"
    checkpoint_dir = base / "checkpoint"
    raw_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame({"id": [1]}).write_parquet(raw_dir / "part-0000.parquet")

    def reader(files):
        return pl.scan_parquet(files)

    def writer(lf, batch_id=None):
        lf.sink_parquet(base / f"out_{batch_id}.parquet")

    pipeline = pli.Pipeline(
        source=pli.FilesSource(path=raw_dir, file_format="parquet"),
        checkpoint_dir=checkpoint_dir,
        reader=reader,
        writer=writer,
    )

    pipeline.run(once=True)

    # Inspect checkpoint state if needed
    assert (checkpoint_dir / "offsets" / "0.json").exists()
    assert (checkpoint_dir / "commits" / "0.json").exists()
```

## Mocking reader/writer

You can return a DataFrame/LazyFrame directly from the reader and use a writer that
records inputs instead of writing to disk.

```python
calls = []

def reader(files):
    return pl.DataFrame({"id": [1], "value": [10]})

def writer(df, batch_id=None):
    calls.append((batch_id, df.height))
    return {"rows": df.height}
```

## Simulate failures and retries

Offsets are written before processing; commits are written only after the writer succeeds.
To test retries, raise once in the writer, then rerun.

```python
fail_once = {"raised": False}

def writer(lf, batch_id=None):
    if not fail_once["raised"]:
        fail_once["raised"] = True
        raise RuntimeError("boom")
    lf.sink_parquet(base / f"out_{batch_id}.parquet")
    return {"ok": True}

pipeline = pli.Pipeline(
    source=pli.FilesSource(path=raw_dir, file_format="parquet"),
    checkpoint_dir=checkpoint_dir,
    reader=reader,
    writer=writer,
)

try:
    pipeline.run(once=True)
except pli.WriterError:
    pass

# Same batch_id is retried
pipeline.run(once=True)
```

## Schema evolution in tests

If you need to validate schema evolution, attach a `SchemaEvolution` config and
write multiple input batches with schema changes. Because the checkpoint stores
the effective schema, keep each test isolated to a fresh checkpoint dir.
