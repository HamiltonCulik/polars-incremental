from __future__ import annotations

import argparse
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


def generate_batch(batch_id: int, rows: int) -> pl.DataFrame:
    now = datetime.now(timezone.utc)
    values = []
    for i in range(rows):
        values.append(
            {
                "event_id": str(uuid.uuid4()),
                "batch_id": batch_id,
                "seq": i,
                "value": round(random.random() * 100, 2),
                "event_ts": now.isoformat(),
            }
        )
    return pl.DataFrame(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed a Delta source table.")
    parser.add_argument("--table", default="data/delta/source_events", help="Delta table path")
    parser.add_argument("--batches", type=int, default=2, help="Number of batches to write")
    parser.add_argument("--rows", type=int, default=5, help="Rows per batch")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing table instead of overwrite on first batch",
    )
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between batches")
    args = parser.parse_args()

    table_path = Path(args.table)
    for batch_id in range(args.batches):
        df = generate_batch(batch_id=batch_id, rows=args.rows)
        mode = "append" if args.append or batch_id > 0 else "overwrite"
        df.write_delta(str(table_path), mode=mode)
        print(f"wrote batch {batch_id} to {table_path} (mode={mode})")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
