# Checkpoints (How They Work)

Most users should never touch checkpoint files directly. The pipeline manages them automatically. This page exists for debugging, inspection, and understanding what the library is doing under the hood.

## Why checkpoints exist

Checkpoints track which inputs were planned and which batches successfully committed. If a batch is planned but not committed, it will be retried on the next run.

## Common layout

All checkpoints share the same top-level layout:

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

- **offsets** record planned input for each batch.
- **commits** record successful completion. A commit is written only after the writer succeeds.
- **metadata.json** stores persistent configuration (e.g., start offsets) and optional schema evolution metadata.

## Planning vs commit

When a batch is planned:
1) An offset file is written.
2) Your reader/transform/writer run.
3) A commit file is written only if the writer succeeds.

If step 2 fails, the offset remains without a commit and that batch is retried.

## File sources

File-based checkpoints track which file paths have been processed.

Additional internal data:

```
checkpoint_dir/
  file_index/
    00.json
    01.json
```

The file index is used for overwrite detection and faster "seen file" checks. It is updated only on successful commits.

## Delta sources

Delta checkpoints track:
- The last processed table version and log index.
- Whether the read was from an initial snapshot or from log changes.

To avoid re-reading the full Delta log every time, we keep a compact snapshot cache:

```
checkpoint_dir/
  snapshot_cache/
    snapshots/
      00000000000000000010.json
    deltas/
      00000000000000000011.json
```

Snapshots represent a compact view of the active file set at a Delta log version.
Deltas store per-version changes since the latest snapshot. The cache is compacted periodically.

If the cache is missing or stale, it can be rebuilt by replaying the Delta log.

## Metadata fields

`metadata.json` can include:
- `start_offset` for persisted start options (files or delta).
- `schema` for schema evolution state (when using `SchemaEvolution`).
- `snapshot_cache` for Delta cache bookkeeping (version and table_id).

## Maintenance helpers

For operational cleanup and debugging:

- `inspect_checkpoint(...)` summarizes offsets/commits and metadata.
- `cleanup_checkpoint(...)` removes old offset/commit files.
- `truncate_checkpoint(...)` truncates offsets/commits after a batch id.
- `cleanup_snapshot_cache(...)` trims Delta snapshot cache files.

See `maintenance.md` and `migrations.md` for usage.
