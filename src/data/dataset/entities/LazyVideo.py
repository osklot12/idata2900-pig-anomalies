from src.data.dataset.entities.video import Video
from src.data.loading.loaders.video_loader import VideoLoader


class LazyVideo(Video):
    """A video that lazy loads the video data."""

    def __init__(self, video_id: str, video_loader: VideoLoader):
        """
        Initializes a LazyVideo instance.

        Args:
            video_id (str): the video id
            video_loader (VideoLoader): the video loader
        """
        self._video_id = video_id
        self._video_loader = video_loader

    def get_id(self) -> str:
        return self._video_id

    def get_data(self) -> bytes:
        return self._video_loader.load_video(self._video_id)