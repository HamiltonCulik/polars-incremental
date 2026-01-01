from __future__ import annotations

import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


def generate_batch(batch_id: int, rows: int = 8) -> pl.DataFrame:
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
    base = Path("data/raw")
    base.mkdir(parents=True, exist_ok=True)

    for batch_id in range(3):
        df = generate_batch(batch_id)
        file_path = base / f"events_{int(time.time())}_{batch_id}.parquet"
        df.write_parquet(file_path)
        print(f"wrote {file_path}")
        time.sleep(0.2)


if __name__ == "__main__":
    main()
