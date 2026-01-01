from pathlib import Path

import polars as pl
import polars_incremental as pli

base_dir = Path("data/maintenance_delta_optimize")
delta_dir = base_dir / "delta" / "events"

pl.DataFrame({"event_id": ["a", "b"], "event_ts": [1, 2]}).write_delta(
    delta_dir, mode="overwrite"
)

# Compact small files (default behavior).
compact_result = pli.optimize_delta_table(
    delta_dir,
    mode="compact",
)
print("compact result:", compact_result)

# Z-order optimization (requires columns).
z_order_result = pli.optimize_delta_table(
    delta_dir,
    mode="z_order",
    z_order_columns=["event_ts", "event_id"],
)
print("z_order result:", z_order_result)
