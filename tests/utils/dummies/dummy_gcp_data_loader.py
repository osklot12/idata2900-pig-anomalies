from typing import List, Dict, Tuple

from src.data.dataset.dataset_source import DatasetSource
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from src.utils.path_finder import PathFinder

class DummyGCPDataLoader(DatasetSource):
    """A fake GCPDataLoader that returns dummy video data."""

    # default constants
    DEFAULT_FRAME_WIDTH = 1920
    DEFAULT_FRAME_HEIGHT = 1080
    DEFAULT_FRAME_COUNT = 171
    DEFAULT_VIDEO_PATH = PathFinder.get_abs_path("tests/data/sample-5s.mp4")
    N_ANNOTATIONS = 8
    DEFAULT_ANNOTATIONS = {
        56: [
            ("g2b_bellynosing", 1925.0824, 1178.3069, 108.3765, 110.6824),
            ("g2b_tailbiting", 1925.0824, 1178.3069, 108.3765, 110.6824),
        ],
        110: [
            ("g2b_bellynosing", 1846.0824, 1203.3069, 107.3742, 89.4320),
            ("g2b_tailbiting", 1925.0824, 1178.3069, 108.3765, 110.6824),
        ],
        128: [
            ("g2b_bellynosing", 1920.4706, 1194.4471, 126.8235, 112.9882),
            ("g2b_tailbiting", 1920.4706, 1194.4471, 126.8235, 112.9882),
        ],
        162: [
            ("g2b_bellynosing", 1920.4706, 1194.4471, 126.8235, 112.9882),
            ("g2b_tailbiting", 1920.4706, 1194.4471, 126.8235, 112.9882),
        ],
    }

    def list_files(self) -> List[str]:
        return self.fetch_all_files()

    def __init__(self, bucket_name: str = "test-bucket", credentials_path: str = "test_credentials.json"):
        self.bucket_name = bucket_name
        self.credentials_path = credentials_path
        self.files = [
            "video1.mp4", "video2.mp4", "video3.mp4",
            "video4.mp4", "video5.mp4", "video6.mp4",
            "video7.mp4", "video8.mp4", "video9.mp4",
        ]
        self.frame_width = DummyGCPDataLoader.DEFAULT_FRAME_WIDTH
        self.frame_height = DummyGCPDataLoader.DEFAULT_FRAME_HEIGHT
        self.frame_count = DummyGCPDataLoader.DEFAULT_FRAME_COUNT
        self.video_path = DummyGCPDataLoader.DEFAULT_VIDEO_PATH
        self.annotations = DummyGCPDataLoader.DEFAULT_ANNOTATIONS

    def download_video(self, blob_name):
        """Returns test video data."""
        return self._get_test_video_data()

    def download_json(self, blob_name):
        """Returns test JSON data."""
        return self._get_test_json_data()

    def fetch_all_files(self):
        """Simulates fetching all available files in the GCP bucket."""
        return self.files

    def get_video_path(self):
        """Returns the video path."""
        return self.video_path

    def set_video_path(self, video_path: str):
        """Sets the video path."""
        self.video_path = PathFinder.get_abs_path(video_path)

    def get_video_properties(self):
        """Returns frame width, height, and count for the video."""
        return self.frame_width, self.frame_height, self.frame_count

    def set_video_properties(self, frame_width: int, frame_height: int, frame_count: int):
        """Sets the video metadata properties dynamically."""
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_count = frame_count

    def get_annotations(self):
        """Returns the annotations."""
        return self.annotations

    def set_annotations(self, annotations: Dict[int, List[Tuple[NorsvinBehaviorClass, float, float, float, float]]]):
        """Sets custom annotations."""
        self.annotations = annotations.copy()

    def _get_test_video_data(self):
        """Generates test video data."""
        with open(self.video_path, "rb") as f:
            video_bytes = f.read()
            print(f"Loaded {len(video_bytes)} bytes from {self.video_path}")
            return bytearray(video_bytes)

    def _get_test_json_data(self):
        """Generates test JSON data."""
        json_data = self._create_json_structure()

        behavior_annotations = self._convert_annotations_to_json()

        # add json annotations to json data
        for behavior, frames_dict in behavior_annotations.items():
            json_data["annotations"].append({
                "name": behavior,
                "frames": frames_dict
            })

        return json_data

    def _convert_annotations_to_json(self):
        """Converts the annotations to JSON format."""
        behavior_annotations = {}
        for frame, annotations in self.annotations.items():
            for behavior, x, y, w, h in annotations:
                if behavior not in behavior_annotations:
                    behavior_annotations[behavior] = {}

                behavior_annotations[behavior][str(frame)] = {
                    "bounding_box": {"x": x, "y": y, "w": w, "h": h},
                    "keyframe": True
                }

        return behavior_annotations

    def _create_json_structure(self):
        """Generates test JSON data structure."""
        return {
            "item": {
                "name": "test_video.mp4",
                "slots": [{
                    "width": self.frame_width,
                    "height": self.frame_height,
                    "frame_count": self.frame_count,
                }]
            },
            "annotations": []
        }