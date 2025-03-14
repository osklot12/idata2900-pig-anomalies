from dataclasses import dataclass

@dataclass(frozen=True)
class DatasetFilePair:
    """Holds a video-annotation file path pair for a dataset in an immutable structure."""
    video_file: str
    annotation_file: str