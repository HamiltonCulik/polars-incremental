from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import polars as pl

from ..cdc import _apply_cdc_prepared, _normalize_change_types, _prepare_changes, apply_cdc

def write_delta(
    df: pl.DataFrame | pl.LazyFrame,
    target: str | Path,
    mode: str = "append",
    *,
    schema_mode: str | None = None,
    collect_kwargs: dict[str, Any] | None = None,
) -> str:
    target_path = Path(target)
    target_path.mkdir(parents=True, exist_ok=True)

    if isinstance(df, pl.LazyFrame):
        collect_kwargs = collect_kwargs or {}
        df = df.collect(**collect_kwargs)

    write_kwargs = {}
    if schema_mode is not None:
        write_kwargs["delta_write_options"] = {"schema_mode": schema_mode}
    df.write_delta(str(target_path), mode=mode, **write_kwargs)
    return "polars"


def apply_cdc_delta(
    df: pl.DataFrame | pl.LazyFrame,
    target: str | Path,
    *,
    keys: Iterable[str],
    change_type_col: str = "_change_type",
    change_type_map: dict[str, str] | None = None,
    mode: str = "merge",
    ignore_delete: bool = False,
    ignore_update_preimage: bool = True,
    dedupe_by_latest_commit: bool = True,
    commit_version_col: str = "_commit_version",
    commit_timestamp_col: str = "_commit_timestamp",
    schema_mode: str | None = None,
    collect_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply CDC rows to a Delta table.

    mode:
      - merge: apply inserts/updates/deletes using keys
      - append_only: append inserts only, ignore deletes/updates
    """
    if isinstance(df, pl.LazyFrame):
        collect_kwargs = collect_kwargs or {}
        df = df.collect(**collect_kwargs)

    if df.is_empty():
        return {"rows_in": 0, "rows_out": 0, "action": "noop"}

    if change_type_map:
        df = _normalize_change_types(df, change_type_col, change_type_map)

    key_list = list(keys)
    if not key_list:
        raise ValueError("keys must include at least one column")
    for key in key_list:
        if key not in df.columns:
            raise ValueError(f"Missing key column: {key}")
    if change_type_col not in df.columns:
        raise ValueError(f"Missing change type column: {change_type_col}")

    change_values = df[change_type_col].unique().to_list()
    if mode not in ("merge", "append_only"):
        raise ValueError(f"Unsupported CDC mode: {mode}")

    df = _prepare_changes(
        df,
        change_type_col=change_type_col,
        ignore_delete=ignore_delete,
        ignore_update_preimage=ignore_update_preimage,
        mode=mode,
    )

    if df.is_empty():
        return {"rows_in": 0, "rows_out": 0, "action": "noop"}

    if mode == "append_only":
        payload = _apply_cdc_prepared(
            df,
            existing=None,
            keys=key_list,
            change_type_col=change_type_col,
            mode=mode,
            dedupe_by_latest_commit=dedupe_by_latest_commit,
            commit_version_col=commit_version_col,
            commit_timestamp_col=commit_timestamp_col,
        )
        existing = _read_delta_if_exists(target)
        if existing is None:
            if payload.is_empty():
                return {"rows_in": df.height, "rows_out": 0, "action": "noop"}
            write_delta(
                payload,
                target,
                mode="overwrite",
                schema_mode=schema_mode,
            )
            return {"rows_in": df.height, "rows_out": payload.height, "action": "append_only"}
        write_delta(
            payload,
            target,
            mode="append",
            schema_mode=schema_mode,
        )
        return {"rows_in": df.height, "rows_out": payload.height, "action": "append_only"}

    existing = _read_delta_if_exists(target)
    if existing is None:
        payload = _apply_cdc_prepared(
            df,
            existing=None,
            keys=key_list,
            change_type_col=change_type_col,
            mode=mode,
            dedupe_by_latest_commit=dedupe_by_latest_commit,
            commit_version_col=commit_version_col,
            commit_timestamp_col=commit_timestamp_col,
        )
        if payload.is_empty():
            return {"rows_in": df.height, "rows_out": 0, "action": "noop"}
        write_delta(
            payload,
            target,
            mode="overwrite",
            schema_mode=schema_mode,
        )
        return {"rows_in": df.height, "rows_out": payload.height, "action": "merge"}

    updated = apply_cdc(
        df,
        existing,
        keys=key_list,
        change_type_col=change_type_col,
        mode=mode,
        ignore_delete=False,
        ignore_update_preimage=False,
        dedupe_by_latest_commit=dedupe_by_latest_commit,
        commit_version_col=commit_version_col,
        commit_timestamp_col=commit_timestamp_col,
    )
    write_delta(
        updated,
        target,
        mode="overwrite",
        schema_mode=schema_mode,
    )
    return {"rows_in": df.height, "rows_out": updated.height, "action": "merge", "change_types": change_values}


def _read_delta_if_exists(target: str | Path) -> pl.DataFrame | None:
    target_path = Path(target)
    if not (target_path / "_delta_log").exists():
        return None
    return pl.read_delta(str(target_path))
