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
        self._id = video_id
        self._loader = video_loader

    def get_id(self) -> str:
        return self._id

    def get_data(self) -> bytes:
        return self._loader.load_video(self._id)