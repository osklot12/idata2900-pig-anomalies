from dataclasses import dataclass

@dataclass(frozen=True)
class DatasetInstance:
    """Represents a single labeled video data instance in a dataset."""
    video_file: str
    annotation_file: str