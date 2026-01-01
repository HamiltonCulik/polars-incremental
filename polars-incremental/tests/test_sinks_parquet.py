import tempfile
import unittest
from pathlib import Path

import polars as pl

from polars_incremental.sinks.parquet import write_parquet_batch


class TestParquetSink(unittest.TestCase):
    def test_write_parquet_batch_dataframe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "out"
            df = pl.DataFrame({"id": [1, 2]})
            result = write_parquet_batch(df, target, batch_id=0)
            self.assertTrue(Path(result["path"]).exists())

    def test_write_parquet_batch_lazyframe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "out"
            df = pl.DataFrame({"id": [1]})
            lf = df.lazy()
            result = write_parquet_batch(lf, target, batch_id=1)
            self.assertTrue(Path(result["path"]).exists())
