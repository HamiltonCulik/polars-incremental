# Advanced patterns

These examples show how to build stateful patterns on top of the checkpointed ingestion loop.
They lean on Polars for transformations and use `JobState` for cross-batch state.

## Incremental aggregations

`examples/advanced-patterns/incremental_aggregations.py`

- Aggregates per batch, then merges into a rolling aggregate table.
- The aggregate table is stored in `data/` and re-read each batch.

## Watermarking

`examples/advanced-patterns/watermarking.py`

- Maintains a simple event-time watermark in `checkpoint/watermark.json`.
- Filters out rows older than `watermark - allowed_lateness`.

## Late data handling

`examples/advanced-patterns/late_data_handling.py`

- Uses the same watermark approach, but splits output into `on_time/` and `late/`.
- Late rows are written to a separate sink for inspection/replay.

## Deduplication strategies

`examples/advanced-patterns/deduplication_strategies.py`

- Strategy A: de-dup by event id using a persisted `seen_ids.json`.
- Strategy B: “latest per user” table updated each batch by event_time.

## State management best practices

### Keep state small
`JobState` is for metadata and small tables. For large state:
- Use Delta tables with versioning
- Store aggregates in a separate database
- Partition state by key ranges

### Handle missing state gracefully
Always provide defaults:
```python
watermark = state.load_json("watermark", default={"value": None})
```

### State is job-specific
Each checkpoint has its own state directory. If you change checkpoint paths,
state won't carry over (by design).

### Clean up obsolete state
For migrations or one-off state entries, remove them explicitly:
```python
if state.exists("watermark_old", kind="json"):
    state.delete("watermark_old", kind="json")
```

### Atomic updates
Writes are atomic. If your job crashes mid-batch, state from the previous
successful batch is preserved.
