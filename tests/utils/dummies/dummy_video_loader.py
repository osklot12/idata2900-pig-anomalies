from src.data.loading.loaders.video_loader import VideoFileLoader


class DummyVideoLoader(VideoFileLoader):
    """Loads a video from disk as raw bytes for testing."""

    def load_video_file(self, video_id: str) -> bytes:
        with open(video_id, 'rb') as f:
            return f.read()