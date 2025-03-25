from src.data.dataset.entities.dataset_file import DatasetFile
from src.data.dataset.entities.video_file import VideoFile
from src.data.loading.loaders.video_file_loader import VideoFileLoader


class LazyVideoFile(VideoFile):
    """A video that lazy loads the video data."""

    def __init__(self, file_path: str, instance_id: str, video_loader: VideoFileLoader):
        """
        Initializes a LazyVideo instance.

        Args:
            file_path (str): the path of the video file
            instance_id (str): the instance ID
            video_loader (VideoFileLoader): the video loader
        """
        super().__init__(file_path, instance_id)
        self._loader = video_loader

    def get_data(self) -> bytes:
        return self._loader.load_video_file(self._file_path)