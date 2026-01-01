"""Example: Catalog-driven ingestion.

Scenario: Use logical dataset names instead of hard-coded paths.
"""

from pathlib import Path

import polars as pl
import polars_incremental as pli

base_dir = Path("data/catalog_basic_example")
raw_dir = base_dir / "raw"
out_dir = base_dir / "out" / "events"
raw_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)

pl.DataFrame({"id": [1, 2], "value": [10, 20]}).write_parquet(raw_dir / "part-0000.parquet")

# Local catalog (JSON/TOML or dict)
local_catalog = {
    "datasets": {
        "raw_events": {"format": "parquet", "path": str(raw_dir)},
        "events_out": {
            "format": "parquet",
            "path": str(out_dir),
        },
    }
}

catalog = pli.LocalCatalog(local_catalog)
source = catalog.get_source("raw_events")
sink = catalog.resolve("events_out")

def reader(files):
    return pl.scan_parquet(files).filter(pl.col("id") > 0)

def writer(lf, batch_id=None):
    Path(sink.path).mkdir(parents=True, exist_ok=True)
    lf.sink_parquet(f"{sink.path}/batch_{batch_id}.parquet")

pipeline = pli.Pipeline(
    source=source,
    checkpoint_dir=base_dir / "checkpoints" / "events_out",
    reader=reader,
    writer=writer,
)

pipeline.run(once=True)

# Databricks-style paths (still just paths; no special integration needed)
dbx_base = base_dir / "dbx"
dbx_bronze = dbx_base / "bronze" / "events"
dbx_silver = dbx_base / "silver" / "events"
dbx_bronze.mkdir(parents=True, exist_ok=True)
dbx_silver.mkdir(parents=True, exist_ok=True)

pl.DataFrame({"event_ts": ["2024-01-01T00:00:00Z"], "id": [1]}).write_delta(
    dbx_bronze, mode="overwrite"
)

dbx_catalog = {
    "datasets": {
        "bronze_events": {
            "format": "delta",
            "path": str(dbx_bronze),
            "ignore_changes": True,
        },
        "silver_events": {
            "format": "delta",
            "path": str(dbx_silver),
        },
    }
}

dbx = pli.LocalCatalog(dbx_catalog)
bronze = dbx.get_source("bronze_events")
silver = dbx.resolve("silver_events")

def dbx_reader(files):
    return pl.scan_parquet(files).with_columns(pl.col("event_ts").cast(pl.Datetime("ms")))

def dbx_writer(lf, batch=None):
    df = lf.collect()
    df.write_delta(silver.path, mode="append")

pipeline = pli.Pipeline(
    source=bronze,
    checkpoint_dir=dbx_base / "checkpoints" / "events_silver",
    reader=dbx_reader,
    writer=dbx_writer,
)

pipeline.run(once=True)

# Unity Catalog-style names mapped to storage locations
uc_base = base_dir / "uc"
uc_bronze_path = uc_base / "bronze" / "events"
uc_silver_path = uc_base / "silver" / "events"
uc_bronze_path.mkdir(parents=True, exist_ok=True)
uc_silver_path.mkdir(parents=True, exist_ok=True)

pl.DataFrame({"event_ts": ["2024-01-02T00:00:00Z"], "id": [2]}).write_delta(
    uc_bronze_path, mode="overwrite"
)

uc_catalog = {
    "datasets": {
        "main.analytics.bronze_events": {
            "format": "delta",
            "path": str(uc_bronze_path),
            "ignore_changes": True,
        },
        "main.analytics.silver_events": {
            "format": "delta",
            "path": str(uc_silver_path),
        },
    }
}

uc = pli.LocalCatalog(uc_catalog)
uc_bronze = uc.get_source("main.analytics.bronze_events")
uc_silver = uc.resolve("main.analytics.silver_events")

def uc_reader(files):
    return pl.scan_parquet(files).with_columns(pl.col("event_ts").cast(pl.Datetime("ms")))

def uc_writer(lf, batch=None):
    df = lf.collect()
    df.write_delta(uc_silver.path, mode="append")

pipeline = pli.Pipeline(
    source=uc_bronze,
    checkpoint_dir=uc_base / "checkpoints" / "events_silver",
    reader=uc_reader,
    writer=uc_writer,
)

pipeline.run(once=True)
