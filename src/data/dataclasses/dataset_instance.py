from dataclasses import dataclass

@dataclass(frozen=True)
class DatasetInstance:
    """
    Represents a single labeled video data instance in a dataset.

    Attributes:
        video_file (str): the path to the video file
        annotation_file (str): the path to the annotation file
    """
    video_file: str
    annotation_file: str