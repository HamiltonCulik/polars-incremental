from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


def write_parquet_batch(
    df: pl.DataFrame | pl.LazyFrame,
    target: str | Path,
    batch_id: int,
    *,
    collect_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target_path = Path(target)
    target_path.mkdir(parents=True, exist_ok=True)
    file_path = target_path / f"batch_{batch_id}.parquet"
    if isinstance(df, pl.LazyFrame):
        if hasattr(df, "sink_parquet"):
            df.sink_parquet(file_path)
        else:
            collect_kwargs = collect_kwargs or {}
            df.collect(**collect_kwargs).write_parquet(file_path)
    else:
        df.write_parquet(file_path)
    return {"path": str(file_path)}
