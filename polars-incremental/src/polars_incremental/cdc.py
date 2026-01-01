from __future__ import annotations

from typing import Any, Iterable

import polars as pl


def apply_cdc(
    changes: pl.DataFrame | pl.LazyFrame,
    existing: pl.DataFrame | pl.LazyFrame | None,
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
    collect_kwargs: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Apply CDC rows to an existing frame (or None) and return the merged result."""
    changes_df = _collect_if_lazy(changes, collect_kwargs)
    existing_df = _collect_if_lazy(existing, collect_kwargs)

    if changes_df.is_empty():
        return existing_df if existing_df is not None else pl.DataFrame()

    if change_type_map:
        changes_df = _normalize_change_types(changes_df, change_type_col, change_type_map)
    key_list = _validate_changes(changes_df, keys, change_type_col)
    if mode not in ("merge", "append_only"):
        raise ValueError(f"Unsupported CDC mode: {mode}")

    changes_df = _prepare_changes(
        changes_df,
        change_type_col=change_type_col,
        ignore_delete=ignore_delete,
        ignore_update_preimage=ignore_update_preimage,
        mode=mode,
    )
    if changes_df.is_empty():
        return existing_df if existing_df is not None else pl.DataFrame()

    return _apply_cdc_prepared(
        changes_df,
        existing_df,
        keys=key_list,
        change_type_col=change_type_col,
        mode=mode,
        dedupe_by_latest_commit=dedupe_by_latest_commit,
        commit_version_col=commit_version_col,
        commit_timestamp_col=commit_timestamp_col,
    )


def _collect_if_lazy(
    df: pl.DataFrame | pl.LazyFrame | None,
    collect_kwargs: dict[str, Any] | None,
) -> pl.DataFrame | None:
    if df is None:
        return None
    if isinstance(df, pl.LazyFrame):
        collect_kwargs = collect_kwargs or {}
        return df.collect(**collect_kwargs)
    return df


def _validate_changes(
    df: pl.DataFrame,
    keys: Iterable[str],
    change_type_col: str,
) -> list[str]:
    key_list = list(keys)
    if not key_list:
        raise ValueError("keys must include at least one column")
    for key in key_list:
        if key not in df.columns:
            raise ValueError(f"Missing key column: {key}")
    if change_type_col not in df.columns:
        raise ValueError(f"Missing change type column: {change_type_col}")
    return key_list


def _prepare_changes(
    df: pl.DataFrame,
    *,
    change_type_col: str,
    ignore_delete: bool,
    ignore_update_preimage: bool,
    mode: str,
) -> pl.DataFrame:
    if ignore_update_preimage:
        df = df.filter(pl.col(change_type_col) != "update_preimage")
    if ignore_delete:
        df = df.filter(pl.col(change_type_col) != "delete")
    if mode == "append_only":
        df = df.filter(pl.col(change_type_col) == "insert")
    return df


def _normalize_change_types(
    df: pl.DataFrame,
    change_type_col: str,
    change_type_map: dict[str, str],
) -> pl.DataFrame:
    if not change_type_map:
        return df
    mapping = dict(change_type_map)
    return df.with_columns(
        pl.col(change_type_col)
        .replace_strict(mapping, default=pl.col(change_type_col))
        .alias(change_type_col)
    )


def _apply_cdc_prepared(
    changes: pl.DataFrame,
    existing: pl.DataFrame | None,
    *,
    keys: list[str],
    change_type_col: str,
    mode: str,
    dedupe_by_latest_commit: bool,
    commit_version_col: str,
    commit_timestamp_col: str,
) -> pl.DataFrame:
    if mode == "append_only":
        if dedupe_by_latest_commit:
            changes = _dedupe_changes(
                changes,
                keys,
                commit_version_col=commit_version_col,
                commit_timestamp_col=commit_timestamp_col,
            )
        payload = _strip_cdc_columns(
            changes,
            change_type_col,
            commit_version_col,
            commit_timestamp_col,
        )
        if existing is None:
            return payload
        return pl.concat([existing, payload], how="diagonal")

    if dedupe_by_latest_commit:
        changes = _dedupe_changes(
            changes,
            keys,
            commit_version_col=commit_version_col,
            commit_timestamp_col=commit_timestamp_col,
        )
        deletes = changes.filter(pl.col(change_type_col) == "delete")
        upserts = changes.filter(
            pl.col(change_type_col).is_in(["insert", "update_postimage", "update"])
        )
    else:
        deletes = changes.filter(pl.col(change_type_col) == "delete").unique(
            subset=keys, keep="last"
        )
        upserts = changes.filter(
            pl.col(change_type_col).is_in(["insert", "update_postimage", "update"])
        )

    if existing is None:
        payload = _strip_cdc_columns(
            upserts,
            change_type_col,
            commit_version_col,
            commit_timestamp_col,
        )
        if deletes.height > 0:
            delete_keys = deletes.select(keys).unique()
            payload = payload.join(delete_keys, on=keys, how="anti")
        return payload

    updated = existing
    if deletes.height > 0:
        delete_keys = deletes.select(keys).unique()
        updated = updated.join(delete_keys, on=keys, how="anti")
    if upserts.height > 0:
        upsert_keys = upserts.select(keys).unique()
        updated = updated.join(upsert_keys, on=keys, how="anti")
        payload = _strip_cdc_columns(
            upserts,
            change_type_col,
            commit_version_col,
            commit_timestamp_col,
        )
        updated = pl.concat([updated, payload], how="diagonal")
    return updated


def _dedupe_changes(
    df: pl.DataFrame,
    keys: list[str],
    *,
    commit_version_col: str,
    commit_timestamp_col: str,
) -> pl.DataFrame:
    order_cols: list[str] = []
    if commit_version_col in df.columns:
        order_cols.append(commit_version_col)
    elif commit_timestamp_col in df.columns:
        order_cols.append(commit_timestamp_col)
    if order_cols:
        df = df.sort(order_cols)
    return df.unique(subset=keys, keep="last")


def _strip_cdc_columns(
    df: pl.DataFrame,
    change_type_col: str,
    commit_version_col: str,
    commit_timestamp_col: str,
) -> pl.DataFrame:
    drop_cols = [change_type_col, commit_version_col, commit_timestamp_col]
    drop_cols = [col for col in drop_cols if col in df.columns]
    return df.drop(drop_cols) if drop_cols else df
