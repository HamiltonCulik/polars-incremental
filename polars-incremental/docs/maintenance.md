# Maintenance helpers

polars-incremental includes a few helpers for managing checkpoints and Delta tables.

Availability note: Delta vacuum/optimize require a recent `deltalake` (delta-rs) install. If `optimize` isn't available, `optimize_delta_table` raises a clear error.

## Checkpoint cleanup

```python
import polars_incremental as pli

result = pli.cleanup_checkpoint(
    "data/checkpoints/events",
    keep_last_n=5,
    older_than_seconds=3600,
    dry_run=True,
)
```

Parameters:
- `keep_last_n` (int): retain N most recent batches.
- `older_than_seconds` (int | None): only delete logs older than this.
- `dry_run` (bool): if true, returns what would be removed.

## Checkpoint migration helpers

```python
import polars_incremental as pli

pli.inspect_checkpoint("data/checkpoints/events")
pli.reset_checkpoint_start_offset("data/checkpoints/events")
pli.reset_checkpoint_schema("data/checkpoints/events")
pli.truncate_checkpoint("data/checkpoints/events", keep_through_batch_id=10)
pli.cleanup_snapshot_cache("data/checkpoints/events", keep_snapshots=2, keep_deltas_since_snapshot=0)
```

See `docs/migrations.md` for when to use each helper safely.

### Snapshot cache cleanup

`cleanup_snapshot_cache` manages `snapshot_cache/` for Delta checkpoint caches:

- `keep_snapshots` (int | None): how many snapshots to retain.
- `keep_deltas_since_snapshot` (int | None): keep deltas newer than the most recent snapshot minus this value.
- `dry_run` (bool): if true, returns what would be removed.

## Delta vacuum

```python
import polars_incremental as pli

result = pli.vacuum_delta_table(
    "data/delta/events",
    retention_hours=168,
    dry_run=True,
    enforce_retention=True,
)
```

Parameters:
- `retention_hours` (float): retention window.
- `dry_run` (bool): if true, only report deletions.
- `enforce_retention` (bool): keep delta-rs safety checks enabled.

Note: delta-rs enforces minimum retention by default. For demos you can disable it:

```python
pli.vacuum_delta_table(
    "data/delta/events",
    retention_hours=0,
    dry_run=False,
    enforce_retention=False,
)
```

## Delta optimize

```python
import polars_incremental as pli

pli.optimize_delta_table("data/delta/events", mode="compact")
pli.optimize_delta_table("data/delta/events", mode="z_order", z_order_columns=["id"])
```

Parameters:
- `mode` (str): `compact` or `z_order`.
- `z_order_columns` (list[str] | None): columns for z-order.
