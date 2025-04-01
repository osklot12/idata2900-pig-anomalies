from enum import Enum, auto

class LogLevel(Enum):
    """Enumeration for log types."""

    INFO = auto()
    """Information logging."""

    WARNING = auto()
    """Warning logging."""

    ERROR = auto()
    """Error logging."""

    DEBUG = auto()
    """Debug logging."""