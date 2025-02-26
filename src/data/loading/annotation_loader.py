import threading
from typing import Callable, Dict, Tuple, Optional, List

from src.data.darwin_decoder import DarwinDecoder
from src.data.loading.feed_status import FeedStatus
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from src.utils.source_normalizer import SourceNormalizer


class AnnotationLoader:
    """
    Loads annotations for videos and invokes callable, feeding the annotations forward.
    """

    def __init__(self, data_loader, callback: Callable[[str, int, Optional[List[Tuple[NorsvinBehaviorClass, float, float, float, float]]], bool], FeedStatus]):
        self.data_loader = data_loader
        self.callback = callback
        self._thread = None

    def load_annotations(self, annotation_blob_name: str):
        """Loads annotations for a video, feeding them to the callback function."""
        self._thread = threading.Thread(target=self._process_annotations, args=(annotation_blob_name,))
        self._thread.start()

    def wait_for_completion(self):
        """Waits for the annotation processing thread to finish."""
        if self._thread:
            self._thread.join()

    def _process_annotations(self, annotation_blob_name: str):
        """Processes annotations and feeds them to the callback function."""
        try:
            annotations_json = self.data_loader.download_json(annotation_blob_name)

            # fetch video name and normalize source id
            video_name = annotations_json.get("item", {}).get("name", "unknown_video")
            source = SourceNormalizer.normalize(video_name)

            # extract image dimensions (width, height)
            slots = annotations_json.get("item")

            frame_count = DarwinDecoder.get_frame_count(annotations_json)
            if frame_count <= 0:
                raise RuntimeError(f"No frames found in {annotation_blob_name}")

            annotations: Dict[int, list] = DarwinDecoder.get_annotations(annotations_json)

            # call callback for every frame
            for frame_index in range(frame_count):
                frame_annotations = annotations.get(frame_index, [])
                print(f"[AnnotationLoader] Processed annotation for frame {frame_index} for source {source}")
                self.callback(source, frame_index, frame_annotations, False)

            # send termination signal
            self.callback(source, None, None, True)
        except Exception as e:
            print(f"Error in _process_annotations: {e}")
