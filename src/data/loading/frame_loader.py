from typing import Callable

import numpy as np
import cv2
from cppbindings import FrameStream

from src.data.dataset_source import DatasetSource
from src.data.loading.frame_loader_interface import FrameLoaderInterface

class FrameLoader(FrameLoaderInterface):
    """
    Loads frames from a video and invokes a callable with video name, frame index, and the frame tensor.
    """

    def __init__(self, data_loader, callback: Callable[[str, int, np.ndarray, bool], None],
                 frame_shape, resize_shape=None):
        self.data_loader = data_loader
        self.callback = callback
        self.frame_shape = frame_shape
        self.resize_shape = resize_shape

    def load_frames(self, video_blob_name: str):
        """Loads frames from a video, feeding them to the callback function."""
        video_data = self.data_loader.download_video(video_blob_name)
        fstream = FrameStream(video_data)

        frame_index = 0
        frame = fstream.read()
        while frame:
            processed_frame = self._process_frame(frame)

            self.callback(video_blob_name, frame_index, processed_frame, False)

            frame = fstream.read()
            frame_index += 1

        # notify that the video processing is complete
        self.callback(video_blob_name, frame_index, None, True)

    def _process_frame(self, frame):
        """Processes a frame by converting it to a np.ndarray and resizing it if necessary."""
        processed_frame = np.frombuffer(bytearray(frame), dtype=np.uint8).reshape(self.frame_shape)
        if self.resize_shape is not None:
            processed_frame = cv2.resize(processed_frame, self.resize_shape)
        return processed_frame

    def get_data_source(self) -> "DatasetSource":
        return self.data_loader
