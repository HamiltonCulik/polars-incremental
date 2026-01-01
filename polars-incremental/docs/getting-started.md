# Getting started

This is the shortest path to a working checkpointed ingestion run.

## Install

From the package root (`polars-incremental/`):

```bash
uv pip install -e .
```

Or in a normal environment:

```bash
pip install polars-incremental
```

## First run (file source)

```python
import polars as pl
import polars_incremental as pli

# 1) create a tiny input file
pl.DataFrame({"id": [1, 2], "value": [10, 20]}).write_parquet("data/raw/part-0000.parquet")

# 2) define reader + writer

def reader(files):
    return pl.scan_parquet(files)

def writer(lf, batch_id=None):
    lf.sink_parquet(f"data/out/batch_{batch_id}.parquet")

# 3) run once
pipeline = pli.Pipeline(
    source=pli.FilesSource(path="data/raw", file_format="parquet"),
    checkpoint_dir="data/checkpoints/out",
    reader=reader,
    writer=writer,
)

pipeline.run(once=True)
```

Re-running the same job will **not** reprocess the file. The checkpoint tracks which inputs have already been committed.

## Next steps

- Learn the checkpoint model: `concepts.md`
- Review the pipeline API: `pipeline.md`
- Explore sources and options: `sources.md`
- Testing pipelines: `testing.md`
- Performance guidance: `performance.md`
