from typing import Callable
import tensorflow as tf
from cppbindings import FrameStream

from src.data.loading.gcp_data_loader import GCPDataLoader


class FrameLoader:
    """
    Loads frames from a video and invokes a callable with video name, frame index, and the frame tensor.
    """
    def __init__(self, bucket_name: str, credentials_path: str, callback: Callable[[str, int, tf.Tensor, bool], None]):
        self.data_loader = GCPDataLoader(bucket_name, credentials_path)
        self.callback = callback

    def load_frames(self, video_blob_name: str):
        video = self.data_loader.download_video(video_blob_name)
        video_bytes = video.getvalue()
        fstream = FrameStream(bytearray(video_bytes))

        frame_index = 0
        frame = fstream.read()
        while frame:
            tf_frame = tf.convert_to_tensor(frame, dtype=tf.uint8)
            self.callback(video_blob_name, frame_index, tf_frame, False)
            frame_index += 1

        # notify that the video processing is complete
        self.callback(video_blob_name, frame_index, None, True)