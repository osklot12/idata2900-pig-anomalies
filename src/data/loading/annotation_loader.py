import threading
from typing import Callable, Dict, Tuple

from src.data.darwin_decoder import DarwinDecoder


class AnnotationLoader:
    """
    Loads annotations for videos and invokes callable, feeding the annotations forward.
    """

    def __init__(self, data_loader, callback: Callable[[str, int, Tuple[str, float, float, float, float], bool], None]):
        self.data_loader = data_loader
        self.callback = callback
        self._thread = None

    def load_annotations(self, video_blob_name: str):
        """Loads annotations for a video, feeding them to the callback function."""
        self._thread = threading.Thread(target=self._process_annotations, args=(video_blob_name,))
        self._thread.start()

    def wait_for_completion(self):
        """Waits for the annotation processing thread to finish."""
        if self._thread:
            self._thread.join()

    def _process_annotations(self, video_blob_name: str):
        """Processes annotations and feeds them to the callback function."""
        annotations_json = self.data_loader.download_json(video_blob_name)

        video_name = annotations_json.get("item", {}).get("name", "unknown_video")
        annotations: Dict[int, list] = DarwinDecoder.decode(annotations_json)

        for frame_index, frame_annotations in annotations.items():
            for annotation in frame_annotations:
                self.callback(video_name, frame_index, annotation, False)

        self.callback(video_name, None, None, True)