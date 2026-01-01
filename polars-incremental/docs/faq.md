# FAQ

## Is this a streaming daemon?

No. polars-incremental is a library that manages checkpoints around your Polars code. Run it under cron/systemd or an orchestrator.

## Why only one writer per checkpoint directory?

The checkpoint is the source of truth for processed inputs. Multiple concurrent writers can race and corrupt progress. Use one writer per checkpoint path.

polars-incremental acquires a per-checkpoint lock when running a pipeline. On Unix it uses an OS file lock; on platforms without `fcntl`, it falls back to a lock file with a timeout. You can disable the lock by setting `POLARS_INCREMENTAL_DISABLE_LOCK=1` or adjust the fallback timeout with `POLARS_INCREMENTAL_LOCK_TIMEOUT`, but the single-writer rule still applies. To recover from stale lock files, set `POLARS_INCREMENTAL_LOCK_STALE_SECONDS` to a positive number; if the lock is older than that threshold or the recorded PID is no longer running, the lock file is cleared and the pipeline proceeds.

## What happens on failure?

Offsets are written before processing a batch, and commits are written only after the writer succeeds. That means:

- If planning fails before an offset is written, no new checkpoint state is created.
- If the reader/transform/writer fails after an offset is written, the commit is not written. On restart, that batch is retried with the same `batch_id` and inputs.
- If the writer succeeds but commit fails, outputs may already exist. Your writer should be idempotent or use `batch_id` in output paths.

See `docs/errors.md` for the error types you can catch.

## Does schema evolution work with LazyFrames?

Yes. Schema rules are applied to both DataFrame and LazyFrame inputs. For LazyFrames, casts and rescue logic are applied lazily and evaluated when collected/sunk.

## Is it safe to retry batches?

polars-incremental provides at-least-once semantics. You should design writers to be idempotent or to tolerate duplicates, especially when a commit fails after the writer already produced output.

## Does polars-incremental manage table scheduling or backfills?

No. Those concerns are left to your scheduler or orchestrator.

## Is the checkpoint stored in the catalog?

No. The catalog only resolves dataset paths and formats. The checkpoint location is job-specific configuration.

## Can I change start offsets or overwrite settings later?

Start offsets and other planning options are persisted in checkpoint metadata on first run. The safest path is to use a new checkpoint directory.

If you need to adjust an existing checkpoint, see `docs/migrations.md` for helper functions like:
- `reset_checkpoint_start_offset(...)`
- `truncate_checkpoint(...)`

When a checkpoint already has a stored start offset and you pass a different one, a warning is logged and the stored value wins.
