from enum import Enum, auto


class DataType(Enum):
    """An enumerations for different dataset instance types."""

    VIDEO = auto()
    """Video data type."""

    ANNOTATION = auto()
    """Annotation data type."""

    UNKNOWN = auto()
    """Unknown data type."""