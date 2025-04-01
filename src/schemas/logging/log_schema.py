from dataclasses import dataclass

from src.schemas.logging.log_level import LogLevel
from src.schemas.schema import Schema


@dataclass(frozen=True)
class LogSchema(Schema):
    """
    A schema for logging.

    Attributes:
        level (LogLevel): the log level
        message (str): the log message
        timestamp (float): the unix timestamp for the event
    """
    level = LogLevel
    message: str
    timestamp: float