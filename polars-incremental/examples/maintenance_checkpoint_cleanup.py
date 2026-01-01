from pathlib import Path
import json

import polars_incremental as pli

base_dir = Path("data/maintenance_checkpoint_cleanup")
checkpoint_dir = base_dir / "checkpoints" / "events_out"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
(checkpoint_dir / "offsets").mkdir(parents=True, exist_ok=True)
(checkpoint_dir / "commits").mkdir(parents=True, exist_ok=True)

for batch_id in range(5):
    (checkpoint_dir / "offsets" / f"{batch_id}.json").write_text(json.dumps({"batch_id": batch_id}))
    (checkpoint_dir / "commits" / f"{batch_id}.json").write_text(json.dumps({"batch_id": batch_id}))

# Keep only the most recent 3 batches in the checkpoint.
result = pli.cleanup_checkpoint(
    checkpoint_dir,
    keep_last_n=3,
)
print("keep_last_n cleanup:", result)

# Remove entries older than 7 days (dry-run to preview).
preview = pli.cleanup_checkpoint(
    checkpoint_dir,
    older_than_seconds=7 * 24 * 60 * 60,
    dry_run=True,
)
print("older_than preview:", preview)
