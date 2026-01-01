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
    parser = argparse.ArgumentParser(description="Append a batch to a Delta source table.")
    parser.add_argument("--table", default="data/delta/source_events", help="Delta table path")
    parser.add_argument("--rows", type=int, default=5, help="Rows per batch")
    parser.add_argument(
        "--batch-id",
        type=int,
        default=None,
        help="Batch id to write (default: unix time)",
    )
    args = parser.parse_args()

    table_path = Path(args.table)
    batch_id = args.batch_id if args.batch_id is not None else int(time.time())
    df = generate_batch(batch_id=batch_id, rows=args.rows)
    df.write_delta(str(table_path), mode="append")
    print(f"appended batch {batch_id} to {table_path}")


if __name__ == "__main__":
    main()
