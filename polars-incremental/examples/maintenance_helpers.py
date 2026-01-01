from pathlib import Path
import json

import polars as pl
import polars_incremental as pli

base_dir = Path("data/maintenance_helpers_example")
checkpoint_dir = base_dir / "checkpoints" / "events_out"
delta_dir = base_dir / "delta" / "events"

checkpoint_dir.mkdir(parents=True, exist_ok=True)
(checkpoint_dir / "offsets").mkdir(parents=True, exist_ok=True)
(checkpoint_dir / "commits").mkdir(parents=True, exist_ok=True)

# Seed a tiny checkpoint for demo cleanup.
for batch_id in range(3):
    (checkpoint_dir / "offsets" / f"{batch_id}.json").write_text(
        json.dumps({"batch_id": batch_id})
    )
    (checkpoint_dir / "commits" / f"{batch_id}.json").write_text(
        json.dumps({"batch_id": batch_id})
    )

# Seed a small Delta table for maintenance demos.
pl.DataFrame({"event_id": ["a", "b"], "event_ts": [1, 2]}).write_delta(
    delta_dir, mode="overwrite"
)

# Cleanup old checkpoint entries (keep only latest 5 batches).
cleanup = pli.cleanup_checkpoint(
    checkpoint_dir,
    keep_last_n=5,
)
print("checkpoint cleanup:", cleanup)

# Vacuum a Delta table (dry-run to preview deletions).
result = pli.vacuum_delta_table(
    delta_dir,
    retention_hours=168.0,
    dry_run=True,
)
print("vacuum result:", result)
