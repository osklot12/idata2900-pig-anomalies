import threading
from typing import Callable

import numpy as np
import cv2
from cppbindings import FrameStream

from src.data.dataset_source import DatasetSource
from src.data.loading.feed_status import FeedStatus
from src.data.loading.streamer import Streamer
from src.utils.source_normalizer import SourceNormalizer


class FrameLoader(Streamer):
    """
    Loads frames from a video and invokes a callable with video name, frame index, and the frame tensor.
    """

    def __init__(self, data_loader, video_blob_name: str, callback: Callable[[str, int, np.ndarray, bool], FeedStatus],
                 frame_shape, resize_shape=None):
        self.data_loader = data_loader
        self.video_blob_name = video_blob_name
        self.callback = callback
        # assumes rgb
        self.frame_shape = (frame_shape[0], frame_shape[1], 3)
        self.resize_shape = resize_shape
        self._thread = None

    def stream(self):
        self._thread = threading.Thread(target=self._process_frames, args=(self.video_blob_name,))
        self._thread.start()

    def wait_for_completion(self):
        if self._thread:
            self._thread.join()

    def _process_frames(self, video_blob_name: str):
        """Processes frames and feeds them to the callback function."""
        video_data = self.data_loader.download_video(video_blob_name)
        source = SourceNormalizer.normalize(video_blob_name)
        fstream = FrameStream(video_data)

        frame_index = 0
        frame = fstream.read()
        while frame:
            print(f"[FrameLoader] Processed frame {frame_index} for source {source}")
            processed_frame = self._process_frame(frame)

            self.callback(source, frame_index, processed_frame, False)

            frame = fstream.read()
            frame_index += 1

        # notify that the video processing is complete
        self.callback(source, frame_index, None, True)

    def _process_frame(self, frame):
        """Processes a frame by converting it to a np.ndarray and resizing it if necessary."""
        processed_frame = np.frombuffer(bytearray(frame), dtype=np.uint8).reshape(self.frame_shape)
        if self.resize_shape is not None:
            processed_frame = cv2.resize(processed_frame, self.resize_shape)
        return processed_frame

    def get_data_source(self) -> "DatasetSource":
        return self.data_loader
