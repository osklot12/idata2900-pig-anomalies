from src.data.loading.loaders.video_loader import VideoLoader


class DummyVideoLoader(VideoLoader):
    """Loads a video from disk as raw bytes for testing."""

    def load_video(self, video_id: str) -> bytes:
        with open(video_id, 'rb') as f:
            return f.read()