class DummyGCPDataLoader:
    """A fake GCPDataLoader that returns dummy video data."""
    def __init__(self, bucket_name, credentials_path):
        self.bucket_name = bucket_name
        self.credentials_path = credentials_path

    def download_video(self, blob_name):
        """Returns a fake video file represented as bytes."""
        return type("DummyVideo", (), {"getValue": lambda: b"fake_video_data"})