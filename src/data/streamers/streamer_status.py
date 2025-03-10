from enum import Enum, auto


class StreamerStatus(Enum):
    """Enumerations for streamer status."""

    PENDING = auto()
    """Streamer is pending."""

    RUNNING = auto()
    """Streamer is running."""

    COMPLETED = auto()
    """Streamer has completed."""

    FAILED = auto()
    """Streamer has failed."""