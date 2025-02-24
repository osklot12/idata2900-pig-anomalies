from src.utils.path_finder import PathFinder

class NorsvinBucketParser:
    """Utility class for handling Norsvin's bucket."""

    CREDENTIALS_PATH = PathFinder.get_abs_path(".secrets/service-account.json")
    BUCKET_NAME = "norsvin-g2b-behavior-prediction"

    VIDEO_PREFIX = "g2b_behaviour/images/"
    JSON_PREFIX = "g2b_behaviour/releases/g2b-prediction/annotations/"

    @staticmethod
    def get_video_blob_name(video_name: str) -> str:
        """Generates the full blob name from the video name."""
        return f"{NorsvinBucketParser.VIDEO_PREFIX}{video_name}"

    @staticmethod
    def get_annotation_blob_name(annotation_name: str) -> str:
        """Generates the full blob name from the annotation name."""
        return f"{NorsvinBucketParser.JSON_PREFIX}{annotation_name}"
