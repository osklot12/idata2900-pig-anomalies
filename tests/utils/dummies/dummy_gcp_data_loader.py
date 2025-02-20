import os
from io import BytesIO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_VIDEO_PATH = os.path.join(BASE_DIR, "../../data/sample-5s.mp4")

class DummyGCPDataLoader:
    """A fake GCPDataLoader that returns dummy video data."""
    def __init__(self, bucket_name, credentials_path):
        self.bucket_name = bucket_name
        self.credentials_path = credentials_path

    def download_video(self, blob_name):
        """Returns test video data."""
        return self._get_test_video_data()

    @staticmethod
    def _get_test_video_data():
        with open(TEST_VIDEO_PATH, "rb") as f:
            video_bytes = f.read()
            print(f"Loaded {len(video_bytes)} bytes from {TEST_VIDEO_PATH}")
            return bytearray(video_bytes)