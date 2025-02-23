from typing import Callable
from cppbindings import FrameStream

from src.data.dataset_source import DatasetSource
from src.data.loading.frame_loader_interface import FrameLoaderInterface


class FrameLoader(FrameLoaderInterface):
    """
    Loads frames from a video and invokes a callable with video name, frame index, and the frame tensor.
    """

    def __init__(self, data_loader, callback: Callable[[str, int, bytearray, bool], None]):
        self.data_loader = data_loader
        self.callback = callback

    def load_frames(self, video_blob_name: str):
        video_data = self.data_loader.download_video(video_blob_name)
        fstream = FrameStream(video_data)

        frame_index = 0
        frame = fstream.read()
        while frame:
            self.callback(video_blob_name, frame_index, frame, False)
            frame = fstream.read()
            frame_index += 1

        # notify that the video processing is complete
        self.callback(video_blob_name, frame_index, None, True)

    def get_data_source(self) -> "DatasetSource":
        return self.data_loader
