# Delta + CDC + CDF

polars-incremental supports Delta as both a **source** and a **sink**. Delta-specific features (Delta sources/sinks, CDC on Delta tables, CDF reads, and Delta maintenance helpers) require `deltalake`.

## Delta sink helper

```python
from polars_incremental.sinks import write_delta

backend = write_delta(
    df_or_lf,
    target="data/delta/events",
    mode="append",
    schema_mode="merge",  # optional
    collect_kwargs={"engine": "streaming"},
)
```

Notes:
- `write_delta` accepts `DataFrame` or `LazyFrame`.
- If given a `LazyFrame`, it will be collected (pass `collect_kwargs` to control engine).
- Return value is a string describing the backend used: `polars`.

Parameters:
- `target` (str or Path): Delta table path.
- `mode` (str): `append` or `overwrite`.
- `schema_mode` (str | None): forwarded to Delta write (e.g. `merge`).
- `collect_kwargs` (dict | None): forwarded to `LazyFrame.collect(...)`.

## CDC apply helper

```python
from polars_incremental.sinks import apply_cdc_delta

result = apply_cdc_delta(
    df_or_lf,
    target="data/delta/events",
    keys=["id"],
    mode="merge",  # or "append_only"
    change_type_col="_change_type",
    collect_kwargs={"engine": "streaming"},
)
```

`apply_cdc_delta` expects change metadata columns (like `_change_type`, `_commit_version`). If a LazyFrame is provided, it will be collected (optional `collect_kwargs`).

If you are not using Delta tables, use the Polars-only `apply_cdc` helper and manage storage yourself (see `examples/cdc_apply.py`).

Important behavior:
- In `append_only` mode, if the table already exists, inserts are appended directly (no full read/overwrite).
- The in-memory merge path is best for small/medium tables or batch-style CDC, not giant tables.

Parameters:
- `keys` (iterable[str]): merge keys (required).
- `change_type_col` (str): change-type column name (default `_change_type`).
- `change_type_map` (dict[str, str] | None): map input change codes (e.g. `{"I": "insert", "U": "update_postimage", "D": "delete"}`).
- `mode` (str): `merge` or `append_only`.
- `ignore_delete` (bool): drop deletes before apply.
- `ignore_update_preimage` (bool): drop update_preimage rows.
- `dedupe_by_latest_commit` (bool): keep last change per key.
- `commit_version_col` / `commit_timestamp_col`: change metadata columns.
- `schema_mode` (str | None): forwarded to Delta write.
- `collect_kwargs` (dict | None): forwarded to `LazyFrame.collect(...)`.

Return value:
- A dict with `rows_in`, `rows_out`, and an `action` key (e.g. `merge`, `append_only`, `noop`).

## CDF reads

If your Delta table has Change Data Feed enabled, you can read CDF batches by setting `read_change_feed=True` in the Delta source.

CDF must be enabled on the Delta table for this to work.

When `read_change_feed=True`, polars-incremental will add change metadata columns (`_change_type`, `_commit_version`, `_commit_timestamp`) if they are missing from the files.

```python
pipeline = pli.Pipeline(
    source=pli.DeltaSource(
        path="data/delta/events",
        read_change_feed=True,
    ),
    checkpoint_dir="data/checkpoints/events",
    reader=reader,
    writer=writer,
)

pipeline.run(once=True)
```
