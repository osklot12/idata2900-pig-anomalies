from enum import Enum, auto

class FeedStatus(Enum):
    """An enumeration that represents the status of a feed."""

    ACCEPT = auto(),
    """The fed instance was accepted."""

    RETRY_LATER = auto(),
    """The fed instance was rejected, but requested to be fed in a retry."""

    DROP = auto()
    """The fed instance was rejected, and requested to be dropped."""