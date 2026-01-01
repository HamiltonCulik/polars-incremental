from __future__ import annotations

from pathlib import Path

import polars as pl
import polars_incremental as pli


def _write_batch(table_path: Path, batch_id: int) -> None:
    df = pl.DataFrame(
        {
            "event_id": [f"e{batch_id}_0", f"e{batch_id}_1"],
            "batch_id": [batch_id, batch_id],
            "value": [batch_id * 1.0, batch_id * 2.0],
        }
    )
    df.write_delta(str(table_path), mode="append")


def main() -> None:
    table_path = Path("data/delta/optimize_demo")
    table_path.mkdir(parents=True, exist_ok=True)

    # Create a few small files to give compaction something to do.
    for batch_id in range(3):
        _write_batch(table_path, batch_id)

    try:
        compact_result = pli.optimize_delta_table(
            table_path,
            mode="compact",
        )
        print("compact result:", compact_result)

        z_order_result = pli.optimize_delta_table(
            table_path,
            mode="z_order",
            z_order_columns=["batch_id"],
        )
        print("z_order result:", z_order_result)
    except RuntimeError as exc:
        print("optimize not available:", exc)


if __name__ == "__main__":
    main()
