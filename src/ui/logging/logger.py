from abc import ABC, abstractmethod

from src.schemas.logging.log_schema import LogSchema


class Logger(ABC):
    """Interface for loggers."""

    @abstractmethod
    def log(self, schema: LogSchema) -> None:
        """
        Logs some information.

        Args:
            schema (LogSchema): the schema to log
        """
        raise NotImplementedError