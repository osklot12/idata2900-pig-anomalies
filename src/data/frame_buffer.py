from collections import deque
from typing import Deque, Tuple

import numpy as np


class FrameBuffer:
    """
    A buffer that stores frames while enforcing a maximum memory limit.
    """

    def __init__(self, max_bytes: int):
        if max_bytes <= 0:
            raise ValueError("Max size must be greater than zero.")

        self.max_bytes = max_bytes
        self.buffer: Deque[Tuple[str, int, np.ndarray]] = deque()
        self.current_mem_usage = 0

    def add_frame(self, video_name: str, frame_index: int, frame: np.ndarray):
        """Adds a new frame to the buffer, removing old frames if necessary."""
        frame_size = frame.nbytes

        if frame_size > self.max_bytes:
            print(
                f"Warning: Frame {frame_index} ({frame_size} bytes) is larger than max buffer size"
                f" ({self.max_bytes} bytes) and will be dropped."
            )
        else:
            self._make_space_for_frame(frame_size)
            self.buffer.append((video_name, frame_index, frame))
            self.current_mem_usage += frame_size

    def _make_space_for_frame(self, frame_size):
        """Removes old frames from the buffer until there is enough space for the new frame."""
        while self.current_mem_usage + frame_size > self.max_bytes and self.buffer:
            _, _, oldest_frame = self.buffer.popleft()
            self.current_mem_usage -= oldest_frame.nbytes

    def get_frames(self):
        """Returns all frames currently in the buffer."""
        return list(self.buffer)

    def __len__(self):
        """Returns the number of frames currently in the buffer."""
        return len(self.buffer)
