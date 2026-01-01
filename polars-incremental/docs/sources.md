# Sources

polars-incremental ships two sources: **files** and **delta**. You configure them with `FilesSource` and `DeltaSource` and pass them into `Pipeline`.

If you want auto-detection, you can still use `SourceSpec(format="auto", ...)` (see `pipeline.md`).

## Files source

```python
source = pli.FilesSource(
    path="data/raw",
    file_format="parquet",  # or "csv" or "excel" or "json" or "ndjson" or "avro"
    pattern="*.parquet",
    recursive=False,
    max_files_per_trigger=100,
    max_bytes_per_trigger=134217728,
    max_file_age=3600.0,
    start_offset="earliest|latest",
    start_timestamp="2024-01-01",
    allow_overwrites=False,
    clean_source="off|delete|archive",
    clean_source_archive_dir="data/archive",
)
```

Options:
- `pattern` (str): glob for input files (default is `*.parquet` unless file format overrides it).
- `recursive` (bool): recursive scan.
- `max_files_per_trigger` (int): cap files per batch.
- `max_bytes_per_trigger` (int): cap batch size in bytes.
- `max_file_age` (float): ignore files older than (latest file mtime - max_file_age) in each scan.
- `start_offset` (str): `earliest` or `latest`.
- `start_timestamp` (str or float): ISO timestamp or unix seconds.
- `allow_overwrites` (bool): detect rewritten files by mtime/size and reprocess them.
- `clean_source` (str): `off` (default), `delete`, or `archive` after commit.
- `clean_source_archive_dir` (str): archive destination when `clean_source="archive"`.
- `file_format` (str): override file format (`parquet`, `csv`, `excel`, `json`, `ndjson`/`jsonl`, or `avro`).

Notes:
- `start_offset` / `start_timestamp` is persisted in checkpoint metadata on first run.
- When `allow_overwrites=True`, file changes are detected via size and mtime.
- If you set `file_format="csv"`, the default pattern is `*.csv` (similarly for other formats).
- `max_file_age` and `clean_source` are **opt-in** retention controls. They can drop late-arriving files if the files are outside the window or already cleaned.
- When `clean_source="archive"` and `recursive=True`, the archive directory under the input path is excluded from scans to avoid re-ingesting archived files. If you want a different location, set `clean_source_archive_dir` outside the input tree.

### Batch sizing and performance

- `max_files_per_trigger` bounds the number of files processed per batch and is the primary lever for controlling memory usage.
- `max_bytes_per_trigger` caps batch size based on file sizes (a single oversized file is still processed).
- Large directories + `recursive=True` increase scan time; prefer narrow globs and non-recursive layouts when possible.
- `allow_overwrites=True` maintains a file index (mtime + size) and adds overhead for large directories.

## Delta source

```python
source = pli.DeltaSource(
    path="data/delta/events",
    start_offset="snapshot",
    starting_version=0,
    starting_timestamp="2024-01-01T00:00:00Z",
    max_files_per_trigger=1000,
    max_bytes_per_trigger=134217728,
    ignore_deletes=False,
    ignore_changes=False,
    read_change_feed=False,
)
```

Options:
- `start_offset` (str): `snapshot` (default), `latest`, or `earliest`.
- `starting_version` (int) / `starting_timestamp` (str): start reading from a Delta version or timestamp.
- `max_files_per_trigger` (int): cap files per batch.
- `max_bytes_per_trigger` (int): cap batch size.
- `ignore_deletes` (bool): ignore delete actions.
- `ignore_changes` (bool): ignore change-only actions.
- `read_change_feed` (bool): read Delta CDF if enabled on the table.

Notes:
- `start_offset` / `starting_version` / `starting_timestamp` is persisted in checkpoint metadata on first run.
- `start_offset="snapshot"` reads the current table snapshot first, then new changes.

### Batch sizing and performance

- `max_files_per_trigger` and `max_bytes_per_trigger` cap the size of each batch.
- For large tables, use `starting_version` or `starting_timestamp` if you only want incremental changes.

## Schema evolution

Schema evolution is configured explicitly via `SchemaEvolution` and attached to the pipeline. See `schema-evolution.md` for details.
