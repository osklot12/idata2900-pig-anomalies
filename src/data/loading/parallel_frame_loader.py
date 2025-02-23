from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Callable, List, Dict

import numpy as np

from src.data.dataset_source import DatasetSource
from src.data.loading.frame_loader import FrameLoader
from src.data.loading.frame_loader_interface import FrameLoaderInterface


class ParallelFrameLoader(FrameLoaderInterface):
    """
    Loads frames from multiple videos in parallel using multiple FrameLoaders.
    """

    def __init__(self, data_loader, callback: Callable[[str, int, np.ndarray, bool], None],
                 frame_shape, resize_shape=None, num_threads = 4):
        self.data_loader = data_loader
        self.callback = callback
        self.frame_shape = frame_shape
        self.resize_shape = resize_shape
        self.num_threads = num_threads
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.futures = []
        self.errors: Dict[str, str] = {}

    def load_frames(self, video_blob_names: List[str]):
        """Loads frames from multiple videos in parallel."""
        for video_name in video_blob_names:
            future = self.executor.submit(self._process_video, video_name)
            self.futures.append((future, video_name))

    def wait_for_completion(self):
        """Waits for all videos to finish processing and reports any errors."""
        for future, video_name in self.futures:
            try:
                future.result()
            except Exception as e:
                self.errors[video_name] = str(e)

        if self.errors:
            print("Some videos failed to process:")
            for video, error, in self.errors.items():
                print(f" - {video}: {error}")

        self.shutdown()

    def _process_video(self, video_name: str):
        """Creates a new FrameLoader and loads frames for a single video."""
        try:
            frame_loader = FrameLoader(self.data_loader, self.callback, self.frame_shape, self.resize_shape)
            frame_loader.load_frames(video_name)
            frame_loader.wait_for_completion()
        except Exception as e:
            self.errors[video_name] = str(e)

    def shutdown(self):
        """Shuts down the thread pool executor."""
        self.executor.shutdown(wait=True)

    def get_data_source(self) -> "DatasetSource":
        return self.data_loader