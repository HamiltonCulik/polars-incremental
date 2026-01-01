class PolarsIncrementalError(Exception):
    """Base error for polars-incremental."""


class PipelineError(PolarsIncrementalError):
    """Base error raised during pipeline execution."""


class PlanningError(PipelineError):
    """Raised when a source fails to plan a batch."""


class CommitError(PipelineError):
    """Raised when a batch commit fails."""


class ReaderError(PipelineError):
    """Raised when the reader callable fails."""


class TransformError(PipelineError):
    """Raised when the transform callable fails."""


class WriterError(PipelineError):
    """Raised when the writer callable fails."""


class UnsupportedFormatError(PolarsIncrementalError):
    """Raised when an unsupported format is requested."""


class MissingOptionError(PolarsIncrementalError):
    """Raised when a required option is missing."""


class ChangeDataFeedError(PolarsIncrementalError):
    """Raised when CDF is required but not available."""


class SchemaEvolutionError(PolarsIncrementalError):
    """Raised when schema evolution rules are violated."""
