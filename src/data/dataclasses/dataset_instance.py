from dataclasses import dataclass

@dataclass(frozen=True)
class DatasetInstance:
    """
    Represents a pair of associated video file and annotations file paths.

    Attributes:
        video_file (str): the path to the video file
        annotation_file (str): the path to the annotation file
    """
    video_file: str
    annotation_file: str