# Job state

polars-incremental does not manage cross-batch state, but it provides a small helper for
persisting state alongside a checkpoint: `JobState`, which you can use to store data
that might be useful to reference between incremental reads.

## Where state lives

For a given checkpoint directory, state is stored under:

```
checkpoint_dir/
  state/
```

This directory is owned by the job using that checkpoint.

## Using JobState in callbacks

If your reader/transform/writer accepts a `state` kwarg, the pipeline will pass a
`JobState` instance automatically.

```python
def transform(df, state=None):
    watermark = state.load_json("watermark", default={"value": None})
    ...
    state.save_json("watermark", {"value": "..."} )
```

## Helper methods

- `load_json(name, default)` / `save_json(name, obj)`
- `load_parquet(name)` / `save_parquet(name, df)`
- `exists(name, kind=None)` / `delete(name, kind=None)`

Writes are atomic (temp file + replace).

## Atomic write guarantee

State writes are crash-safe at the file level: the previous value remains until the new
file is fully written and replaced. If a job crashes mid-write, you keep the last
successful state.

## When to use it

- Watermarks
- Dedupe state (seen ids)
- Rolling aggregates or “latest per key” tables

See `docs/advanced-patterns.md` for end-to-end examples.

## Cleanup example

```python
if state.exists("watermark_old", kind="json"):
    state.delete("watermark_old", kind="json")
```
