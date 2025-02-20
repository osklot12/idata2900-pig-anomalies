from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Callable, List

from src.data.loading.frame_loader import FrameLoader

class ParallelFrameLoader:
    """
    Loads frames from multiple videos in parallel using multiple FrameLoaders.
    """

    def __init__(self, data_loader, callback: Callable[[str, int, bytearray, bool], None], num_threads = 4):
        self.data_loader = data_loader
        self.callback = callback
        self.num_threads = num_threads

    def load_frames(self, video_blob_names: List[str]):
        """Loads frames from multiple videos in parallel."""
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_video = {
                executor.submit(FrameLoader(self.data_loader, self.callback).load_frames, video): video
                for video in video_blob_names
            }

            for future in as_completed(future_to_video):
                video_name = future_to_video[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing video {video_name}: {e}")
