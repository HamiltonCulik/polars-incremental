import logging
import unittest

from polars_incremental.observability import LoggingObserver


class TestLoggingObserver(unittest.TestCase):
    def test_logging_observer_events(self) -> None:
        logger = logging.getLogger("polars_incremental.test")
        observer = LoggingObserver(logger)

        class Batch:
            batch_id = 7

        with self.assertLogs("polars_incremental.test", level="INFO") as captured:
            observer.on_batch_planned(Batch(), ["a.parquet", "b.parquet"])
            observer.on_stage_start("reader", 7)
            observer.on_stage_end("reader", 7, 0.01, metadata={"rows": 2})
            observer.on_batch_committed(7, metadata={"ok": True})

        joined = "\n".join(captured.output)
        self.assertIn("event=batch_planned", joined)
        self.assertIn("event=stage_start", joined)
        self.assertIn("event=stage_end", joined)
        self.assertIn("event=batch_committed", joined)

    def test_logging_observer_error(self) -> None:
        logger = logging.getLogger("polars_incremental.test_error")
        observer = LoggingObserver(logger)
        exc = ValueError("boom")

        with self.assertLogs("polars_incremental.test_error", level="ERROR") as captured:
            observer.on_error("reader", 1, exc)

        self.assertTrue(any("event=error" in line for line in captured.output))
