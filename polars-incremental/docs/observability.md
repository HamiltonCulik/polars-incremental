# Observability and logging

polars-incremental includes an optional observer interface so you can emit structured logs
or metrics without wrapping your reader/writer manually.

## Using the logging observer

```python
import logging
import polars_incremental as pli

logger = logging.getLogger("polars_incremental")
logger.setLevel(logging.INFO)

pipeline = pli.Pipeline(
    source=pli.FilesSource(path="data/raw", file_format="parquet"),
    checkpoint_dir="data/checkpoints/raw",
    reader=reader,
    writer=writer,
    observer=pli.LoggingObserver(logger),
)

pipeline.run(once=True)
```

The logging observer emits simple key=value messages such as:

```
event=stage_start stage=reader batch_id=0
event=stage_end stage=reader batch_id=0 duration_s=0.013 file_count=10
event=batch_committed batch_id=0
```

See `examples/observability_basic.py` for a minimal setup.

## Custom observers

Implement the `PipelineObserver` protocol to hook into lifecycle events:

```python
class MyObserver:
    def on_batch_planned(self, batch, files):
        ...
    def on_stage_start(self, stage, batch_id):
        ...
    def on_stage_end(self, stage, batch_id, duration_s, metadata=None):
        ...
    def on_batch_committed(self, batch_id, metadata=None):
        ...
    def on_error(self, stage, batch_id, exc):
        ...
```

Notes:
- `metadata` is whatever your writer returns (if it returns a dict).
- The pipeline does **not** compute row counts for you; include them in your writer metadata if desired.
