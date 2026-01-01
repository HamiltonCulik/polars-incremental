# Concepts

## Checkpoints

polars-incremental writes Spark-style checkpoints to a directory you provide:

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

- **Offsets** store the planned inputs for a batch.
- **Commits** are written only after your writer succeeds.
- If an offset exists without a commit, the batch is retried on restart.

This yields **at-least-once** processing. Your writer should be idempotent or able to tolerate retries.

If you need to change source options that affect planning (start offsets, overwrite behavior, etc.), use a new checkpoint directory. The initial planning config is persisted in checkpoint metadata.

Tip: use `batch_id` in output paths (e.g. `batch_{batch_id}.parquet`) to make retries safe.

For checkpoint internals and cache details, see `checkpoints.md`.

## Batches

Each run plans a batch of inputs (files or Delta log entries). That batch is passed to your `reader(files)` which returns a Polars DataFrame or LazyFrame.

## Scheduling

polars-incremental is a library, not a daemon. You run `Pipeline.run(...)` from:

- Cron/systemd
- Airflow/Prefect/Dagster
- A long-running process you manage

## Schema evolution

Schema evolution is optional. When enabled, polars-incremental compares each batch to the stored schema and enforces the policy you choose. The schema is persisted in checkpoint metadata to keep behavior stable across restarts.

## Sources

Two built-in sources:

- **Files**: incremental scans with optional overwrite detection and start offsets.
- **Delta**: incremental reads based on the Delta log (with optional CDF).

Details and options: `sources.md`.

## CDC helpers

polars-incremental includes two CDC helpers with different scopes:

- `apply_cdc_delta` applies CDC rows to a **Delta table** (Delta-specific I/O).
- `apply_cdc` applies CDC rows to an **in-memory Polars DataFrame**; you handle storage yourself (e.g., write to parquet/DB).

Both helpers accept `change_type_map` to normalize custom change-type codes (e.g., `{"I": "insert", "U": "update_postimage", "D": "delete"}`).

See `examples/cdc_apply.py` and `examples/cdc_apply_delta.py` for minimal usage.
