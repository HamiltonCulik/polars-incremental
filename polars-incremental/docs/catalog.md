# Catalog

The catalog is optional. It provides a lightweight **name -> dataset spec** resolver so you can centralize paths, formats, and options.

## What you get from `resolve(...)`

`catalog.resolve(name)` returns a `DatasetSpec` with these fields:

- `name`: dataset name (string)
- `format`: dataset format (string, e.g. `parquet`, `delta`)
- `path`: dataset path (string or `Path`)
- `options`: extra options (dict)

`DatasetSpec.to_source()` converts a spec into a `FilesSource` or `DeltaSource`.

## Minimal usage

```python
import polars as pl
import polars_incremental as pli

catalog = pli.LocalCatalog({
    "datasets": {
        "raw_events": {"format": "parquet", "path": "data/raw"},
        "events_out": {"format": "parquet", "path": "data/out/events"},
    }
})

source = catalog.get_source("raw_events")
sink = catalog.resolve("events_out")


def reader(files):
    return pl.scan_parquet(files)


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

The checkpoint path is still a job configuration, not catalog metadata.

## Using catalog options

`DatasetSpec.options` is a free-form dict meant to hold **source-specific options** that you want to reuse across jobs. It helps keep ingestion behavior (patterns, start offsets, CDF, etc.) consistent without repeating those settings in every job. Schema evolution settings belong in `options` only if you also call `spec.to_schema_evolution()` and pass the result into `Pipeline`.

Examples of options that belong here:
- File source: `pattern`, `max_files_per_trigger`, `start_offset`, `allow_overwrites`
- Delta source: `starting_version`, `read_change_feed`, `max_bytes_per_trigger`

You typically **do not** put job-specific settings here (like `checkpoint_dir`, `run_once`, or `sleep`). Those stay in the job config.

A common pattern is to resolve the spec into a source and schema evolution config:

```python
spec = catalog.resolve("raw_events")
source = spec.to_source()
schema_evolution = spec.to_schema_evolution()

pli.Pipeline(
    source=source,
    checkpoint_dir="data/checkpoints/raw_events",
    reader=reader,
    writer=writer,
    schema_evolution=schema_evolution,
).run(once=True)
```

## File-based catalog

`LocalCatalog` can load JSON or TOML files. The top level can be either a direct dataset mapping or a `datasets` key.

Example JSON:

```json
{
  "datasets": {
    "raw_events": {
      "format": "parquet",
      "path": "data/raw",
      "pattern": "events_*.parquet",
      "max_files_per_trigger": 100
    }
  }
}
```

Extra keys become `DatasetSpec.options`.
