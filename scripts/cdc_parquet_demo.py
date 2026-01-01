from __future__ import annotations

from pathlib import Path

import polars as pl
import polars_incremental as pli


def main() -> None:
    base_dir = Path("data/cdc_parquet_demo")
    base_dir.mkdir(parents=True, exist_ok=True)
    existing_path = base_dir / "existing.parquet"
    changes_path = base_dir / "changes.parquet"
    output_path = base_dir / "output.parquet"

    existing = pl.DataFrame({"id": [1, 2], "value": [10, 20]})
    existing.write_parquet(existing_path)

    changes = pl.DataFrame(
        {
            "id": [2, 1, 3],
            "value": [25, None, 30],
            "_change_type": ["update_postimage", "delete", "insert"],
            "_commit_version": [1, 2, 3],
        }
    )
    changes.write_parquet(changes_path)

    changes_df = pl.read_parquet(changes_path)
    existing_df = pl.read_parquet(existing_path)
    updated = pli.apply_cdc(changes_df, existing_df, keys=["id"])
    updated.write_parquet(output_path)

    print("existing:")
    print(existing_df.sort("id"))
    print("changes:")
    print(changes_df.sort("id"))
    print("updated:")
    print(pl.read_parquet(output_path).sort("id"))


if __name__ == "__main__":
    main()
