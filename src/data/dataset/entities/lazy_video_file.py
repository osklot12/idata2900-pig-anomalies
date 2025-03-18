from src.data.dataset.entities.video_file import VideoFile
from src.data.loading.loaders.video_file_loader import VideoFileLoader


class LazyVideoFile(VideoFile):
    """A video that lazy loads the video data."""

    def __init__(self, video_id: str, video_loader: VideoFileLoader):
        """
        Initializes a LazyVideo instance.

        Args:
            video_id (str): the video id
            video_loader (VideoFileLoader): the video loader
        """
        self._id = video_id
        self._loader = video_loader

    def get_id(self) -> str:
        return self._id

    def get_data(self) -> bytes:
        return self._loader.load_video_file(self._id)