from dataclasses import dataclass

from src.schemas.logging.log_level import LogLevel
from src.schemas.schemas.timestamped_schema import TimestampedSchema


@dataclass(frozen=True)
class LogSchema(TimestampedSchema):
    """
    A schema for logging.

    Attributes:
        level (LogLevel): the log level
        message (str): the log message
    """
    level: LogLevel
    message: str