from enum import Enum, auto


class DataType(Enum):
    """An enumerations for different dataset instance types."""

    VIDEO = auto()
    """Video data type."""

    ANNOTATION = auto()
    """Annotation data type."""

    UNKNOWN = auto()
    """Unknown data type."""


    def flip(self) -> "DataType":
        """Returns VIDEO if input is ANNOTATION, ANNOTATION if input is VIDEO."""
        return {
            DataType.VIDEO: DataType.ANNOTATION,
            DataType.ANNOTATION: DataType.VIDEO
        }.get(self, DataType.UNKNOWN)