# Performance guidance

This page summarizes the main knobs that affect throughput and memory usage.

## Checkpoint overhead

Checkpointing writes one JSON offset and one JSON commit per batch. Planning a batch
requires listing committed batches and reading their offsets to discover which files
have already been processed.

For long-running jobs with many batches, consider periodic checkpoint cleanup to
keep planning fast.

## File source tuning

- `max_files_per_trigger` bounds the number of files read per batch and is the
  primary lever for memory usage and batch size.
- `max_bytes_per_trigger` caps batch size by file size; a single oversized file
  is still included so progress can be made.
- `recursive=True` increases filesystem scanning cost. Prefer narrow globs and
  non-recursive layouts when possible.
- `allow_overwrites=True` maintains a file index (mtime + size) and may add overhead
  when scanning large directories.

If you are reading many small files, use `pl.scan_*` and perform collection in the
writer to control memory usage.

## Delta source tuning

- `max_files_per_trigger` and `max_bytes_per_trigger` cap the batch size produced by
  the Delta log planner.
- `start_offset="snapshot"` reads the table snapshot first; use `starting_version`
  when you only want incremental changes.

## LazyFrame execution control

`Pipeline` never calls `collect()` for you. If your reader returns a `LazyFrame`, you
control execution in your transform or writer:

- Use `collect(engine="streaming")` for lower memory usage.
- Use `sink_parquet` / `sink_csv` for fully streaming writes.

This lets you pick the right trade-offs for batch size and resource constraints.

## Backpressure behavior

`Pipeline` is single-threaded and processes **one batch at a time**. If your writer
is slow, the loop naturally slows down because the next batch is not planned until
the writer returns.

For long-running jobs, use smaller batches (`max_files_per_trigger` /
`max_bytes_per_trigger`) and streaming collection in the writer.
