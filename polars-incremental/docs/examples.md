# Examples index

These examples live under `polars-incremental/examples/` and are runnable as-is.
Advanced patterns are under `polars-incremental/examples/advanced-patterns/`.

- `basic_checkpointing.py` — minimal file -> checkpoint -> sink.
- `schema_evolution.py` — schema evolution policies.
- `schema_evolution_demo.py` — schema evolution with rescue.
- `schema_type_widen.py` — type widening behavior.
- `files_start_offset.py` — start offsets for file source.
- `file_overwrites.py` — allow_overwrites behavior.
- `delta_schema_evolution_write.py` — Delta write + schema evolution.
- `delta_cdf.py` — CDF read.
- `cdc_apply.py` — Polars-only CDC merge (no Delta I/O).
- `cdc_apply_delta.py` — CDC apply helper for Delta tables.
- `catalog_basic.py` — Local catalog usage.
- `maintenance_helpers.py` — checkpoint cleanup + vacuum + optimize.
- `maintenance_checkpoint_cleanup.py` — checkpoint cleanup demo.
- `maintenance_delta_vacuum.py` — Delta vacuum demo.
- `maintenance_delta_optimize.py` — Delta optimize demo.
- `checkpoint_migration.py` — truncate checkpoint and reprocess a batch.
- `observability_basic.py` — logging observer example.
- `advanced-patterns/incremental_aggregations.py` — rolling aggregates across batches.
- `advanced-patterns/watermarking.py` — watermark-based filtering with manual state.
- `advanced-patterns/late_data_handling.py` — route late rows to a separate sink.
- `advanced-patterns/deduplication_strategies.py` — id-dedupe and latest-per-user patterns.
