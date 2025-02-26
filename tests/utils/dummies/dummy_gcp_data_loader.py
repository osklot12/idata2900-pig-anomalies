import json
import os
from typing import List

from src.data.dataset_source import DatasetSource

# directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_VIDEO_PATH = os.path.join(BASE_DIR, "../../data/sample-5s.mp4")


class DummyGCPDataLoader(DatasetSource):
    """A fake GCPDataLoader that returns dummy video data."""

    # constants
    FRAME_WIDTH = 1920
    FRAME_HEIGHT = 1080
    FRAME_COUNT = 171
    N_ANNOTATIONS = 4
    ANNOTATIONS = [
        ("g2b_bellynosing", [
            # frame, x, y, w, h
            ("56", 1925.0824, 1178.3069, 108.3765, 110.6824),
            ("110", 1846.0824, 1203.3069, 107.3742, 89.4320),
        ]),
        ("g2b_bellynosing", [
            ("87", 1654.0443, 478.1234, 87.3424, 130.4320),
            ("110", 1324.8329, 804.8342, 101.3482, 102.7300),
        ])
    ]

    def list_files(self) -> List[str]:
        return self.fetch_all_files()

    def __init__(self, bucket_name: str = "test-bucket", credentials_path: str = "test_credentials.json"):
        self.bucket_name = bucket_name
        self.credentials_path = credentials_path
        self.dummy_json = self._get_test_json_data()
        self.files = [
            "video1.mp4", "video2.mp4", "video3.mp4",
            "video4.mp4", "video5.mp4", "video6.mp4",
            "video7.mp4", "video8.mp4", "video9.mp4",
        ]

    def download_video(self, blob_name):
        """Returns test video data."""
        return self._get_test_video_data()

    def download_json(self, blob_name):
        """Returns test JSON data."""
        return self.dummy_json

    def fetch_all_files(self):
        """Simulates fetching all available files in the GCP bucket."""
        return self.files

    def set_dummy_json(self, json_data):
        """Sets the dummy JSON data."""
        self.dummy_json = json.loads(json_data)

    @staticmethod
    def _get_test_video_data():
        """Generates test video data."""
        with open(TEST_VIDEO_PATH, "rb") as f:
            video_bytes = f.read()
            print(f"Loaded {len(video_bytes)} bytes from {TEST_VIDEO_PATH}")
            return bytearray(video_bytes)

    @staticmethod
    def _get_test_json_data():
        """Generates test JSON data."""
        # construct the JSON dictionary
        json_data = {
            "item": {
                "name": "test_video.mp4",
                "slots": [{
                    "width": DummyGCPDataLoader.FRAME_WIDTH,
                    "height": DummyGCPDataLoader.FRAME_HEIGHT,
                    "frame_count": DummyGCPDataLoader.FRAME_COUNT
                }]
            },
            "annotations": []
        }

        # iterate over all annotations and add them dynamically
        for behavior, frames in DummyGCPDataLoader.ANNOTATIONS:
            frames_dict = {
                frame: {
                    "bounding_box": {
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h
                    },
                    "keyframe": True
                } for frame, x, y, w, h in frames
            }

            json_data["annotations"].append({
                "name": behavior,
                "frames": frames_dict
            })

        return json_data
