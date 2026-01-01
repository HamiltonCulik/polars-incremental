from __future__ import annotations

import logging
from typing import Any, Protocol


class PipelineObserver(Protocol):
    def on_batch_planned(self, batch: Any, files: list[str]) -> None:
        raise NotImplementedError

    def on_stage_start(self, stage: str, batch_id: int | None) -> None:
        raise NotImplementedError

    def on_stage_end(
        self,
        stage: str,
        batch_id: int | None,
        duration_s: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    def on_batch_committed(self, batch_id: int | None, metadata: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

    def on_error(self, stage: str, batch_id: int | None, exc: Exception) -> None:
        raise NotImplementedError


class LoggingObserver:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("polars_incremental")

    def on_batch_planned(self, batch: Any, files: list[str]) -> None:
        self._log(
            "batch_planned",
            batch_id=getattr(batch, "batch_id", None),
            file_count=len(files),
        )

    def on_stage_start(self, stage: str, batch_id: int | None) -> None:
        self._log("stage_start", stage=stage, batch_id=batch_id)

    def on_stage_end(
        self,
        stage: str,
        batch_id: int | None,
        duration_s: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {"stage": stage, "batch_id": batch_id, "duration_s": duration_s}
        if metadata:
            payload["metadata"] = metadata
        self._log("stage_end", **payload)

    def on_batch_committed(self, batch_id: int | None, metadata: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"batch_id": batch_id}
        if metadata:
            payload["metadata"] = metadata
        self._log("batch_committed", **payload)

    def on_error(self, stage: str, batch_id: int | None, exc: Exception) -> None:
        self._logger.error(
            "event=error stage=%s batch_id=%s error=%s",
            stage,
            batch_id,
            exc,
            exc_info=exc,
        )

    def _log(self, event: str, **fields: Any) -> None:
        parts = [f"event={event}"]
        for key, value in fields.items():
            parts.append(f"{key}={value}")
        self._logger.info(" ".join(parts))
