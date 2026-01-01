# Migration and checkpoint changes

This page describes how to safely change planning options or schema behavior for an existing checkpoint.

## The rule of thumb

Planning options (start offsets, timestamps, file overwrite tracking) are persisted in
checkpoint metadata on first run. If you need different planning behavior, the safest
option is to **use a new checkpoint directory**.

## Inspecting a checkpoint

Use `inspect_checkpoint` to summarize state and metadata:

```python
import polars_incremental as pli

info = pli.inspect_checkpoint("data/checkpoints/events")
print(info)
```

`CheckpointInfo` includes counts, latest batch ids, pending batches, and stored metadata.

## Resetting start offsets

If you want to re-run with a different start offset on the **same checkpoint**, you can
clear the persisted value:

```python
import polars_incremental as pli

pli.reset_checkpoint_start_offset("data/checkpoints/events")
```

On the next run, the source will apply the new start settings and persist them.

## Resetting schema evolution state

Schema evolution stores the effective schema in checkpoint metadata. To clear it:

```python
import polars_incremental as pli

pli.reset_checkpoint_schema("data/checkpoints/events")
```

Use this only if you are intentionally re-baselining schema tracking.

## Truncating checkpoint history

If a batch is bad or you want to roll back to a previous batch id, you can truncate:

```python
import polars_incremental as pli

pli.truncate_checkpoint("data/checkpoints/events", keep_through_batch_id=10)
```

This removes **offsets and commits** with batch ids greater than the threshold. The next
run will plan and reprocess batches starting at the next id.

## Recommended migration recipes

1. **Change start offsets / planning logic**: create a new checkpoint dir (safest).
2. **Re-baseline schema evolution**: reset schema metadata.
3. **Reprocess after a bad batch**: truncate checkpoint and rerun.

See `examples/checkpoint_migration.py` for an end-to-end example.
