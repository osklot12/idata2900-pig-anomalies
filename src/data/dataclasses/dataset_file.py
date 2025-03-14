from dataclasses import dataclass

from src.data.dataset.data_type import DataType


@dataclass(frozen=True)
class DatasetFile:
    """Holds dataset file information in an immutable structure."""
    file_path: str
    type: DataType