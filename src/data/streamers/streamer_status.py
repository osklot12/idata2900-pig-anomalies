from enum import Enum, auto


class StreamerStatus(Enum):
    """Enumerations for streamer status."""

    PENDING = auto()
    """Streamer has not started yet."""

    STREAMING = auto()
    """Streamer is actively streaming."""

    STOPPED = auto()
    """Streamer was manually stopped."""

    COMPLETED = auto()
    """Streamer has finished naturally."""

    FAILED = auto()
    """Streamer has encountered an error."""