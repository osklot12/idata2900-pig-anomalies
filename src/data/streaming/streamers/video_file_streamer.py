from typing import Optional

import numpy as np
from cppbindings import FrameStream

from src.data.dataclasses.frame import Frame
from src.data.dataclasses.source_metadata import SourceMetadata
from src.data.dataset.entities.video_file import VideoFile
from src.data.preprocessing.resizing.resizers.frame_resizer import FrameResizer
from src.data.pipeline.consumer import Consumer
from src.data.streaming.streamers.video_streamer import VideoStreamer


class VideoFileStreamer(VideoStreamer):
    """A streamer for streaming video file data."""

    def __init__(self, video: VideoFile, consumer: Consumer[Frame]):
        """
        Initializes a VideoFileStreamer instance.

        Args:
            video (VideoFile): the video to streams
            consumer (Consumer[Frame]): the consumer of the streaming data
        """
        super().__init__(consumer)
        self._video = video
        self._fstream = None
        self._frame_index = 0

    def _setup_stream(self) -> None:
        video_data = self._video.get_data()
        self._fstream = FrameStream(bytearray(video_data))

    def _get_next_frame(self) -> Optional[Frame]:
        frame = None

        frame_data = self._fstream.read()
        if frame_data:
            width, height, channels = self._fstream.get_frame_shape()
            np_data = np.frombuffer(frame_data, dtype=np.uint8)
            np_data = np_data.reshape((height, width, channels))

            source_meta = SourceMetadata(self._video.get_instance_id(), (width, height))
            frame = Frame(source_meta, self._frame_index, np_data)

            self._frame_index += 1

        return frame