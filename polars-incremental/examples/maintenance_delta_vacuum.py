from pathlib import Path

import polars as pl
import polars_incremental as pli

base_dir = Path("data/maintenance_delta_vacuum")
delta_dir = base_dir / "delta" / "events"

pl.DataFrame({"event_id": ["a", "b"], "event_ts": [1, 2]}).write_delta(
    delta_dir, mode="overwrite"
)

# Vacuum a Delta table with a 7-day retention window (dry-run).
preview = pli.vacuum_delta_table(
    delta_dir,
    retention_hours=168.0,
    dry_run=True,
)
print("vacuum preview:", preview)

# Real vacuum example (uncomment when ready):
# pli.vacuum_delta_table(delta_dir, retention_hours=168.0)
