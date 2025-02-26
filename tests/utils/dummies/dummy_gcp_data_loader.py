import json
import os
from typing import List

from src.data.dataset_source import DatasetSource

# directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_VIDEO_PATH = os.path.join(BASE_DIR, "../../data/sample-5s.mp4")


class DummyGCPDataLoader(DatasetSource):
    """A fake GCPDataLoader that returns dummy video data."""

    def list_files(self) -> List[str]:
        return self.fetch_all_files()

    def __init__(self, bucket_name, credentials_path):
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
        json_data = """
            {
                "item": {
                    "name": "test_video.mp4",
                    "slots": [
                        {
                            "frame_count": 171
                        }
                    ]
                },
                "annotations": [
                    {
                        "name": "g2b_bellynosing",
                        "frames": {
                            "56": {
                                "bounding_box": {
                                    "x": 1925.0824,
                                    "y": 1178.3059,
                                    "w": 108.3765,
                                    "h": 110.6824
                                },
                                "keyframe": true
                            },
                            "110": {
                                "bounding_box": {
                                    "x": 1925.0824,
                                    "y": 1178.3059,
                                    "w": 108.3765,
                                    "h": 110.6824
                                },
                                "keyframe": true
                            },
                            "128": {
                                "bounding_box": {
                                    "x": 1920.4706,
                                    "y": 1194.4471,
                                    "w": 126.8235,
                                    "h": 112.9882
                                },
                                "keyframe": true
                            },
                            "162": {
                                "bounding_box": {
                                    "x": 1920.4706,
                                    "y": 1194.4471,
                                    "w": 126.8235,
                                    "h": 112.9882
                                },
                                "keyframe": true
                            }
                        }
                    },
                    {
                        "name": "g2b_tailbiting",
                        "frames": {
                            "56": {
                                "bounding_box": {
                                    "x": 1925.0824,
                                    "y": 1178.3059,
                                    "w": 108.3765,
                                    "h": 110.6824
                                },
                                "keyframe": true
                            },
                            "110": {
                                "bounding_box": {
                                    "x": 1925.0824,
                                    "y": 1178.3059,
                                    "w": 108.3765,
                                    "h": 110.6824
                                },
                                "keyframe": true
                            },
                            "128": {
                                "bounding_box": {
                                    "x": 1920.4706,
                                    "y": 1194.4471,
                                    "w": 126.8235,
                                    "h": 112.9882
                                },
                                "keyframe": true
                            },
                            "162": {
                                "bounding_box": {
                                    "x": 1920.4706,
                                    "y": 1194.4471,
                                    "w": 126.8235,
                                    "h": 112.9882
                                },
                                "keyframe": true
                            }
                        }
                    }
                ]
            }
            """
        return json.loads(json_data)
