import threading
from typing import Callable, Dict, Tuple, Optional, List

from src.data.darwin_decoder import DarwinDecoder
from src.data.loading.feed_status import FeedStatus
from src.utils.annotation_normalizer import AnnotationNormalizer
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from src.utils.source_normalizer import SourceNormalizer


class AnnotationLoader:
    """
    Loads annotations for video streams and invokes callable, feeding the annotations forward.
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
            original_width, original_height = DarwinDecoder.get_frame_dimensions(annotations_json)

            frame_count = DarwinDecoder.get_frame_count(annotations_json)
            if frame_count <= 0:
                raise RuntimeError(f"No frames found in {annotation_blob_name}")

            annotations: Dict[int, list] = DarwinDecoder.get_annotations(annotations_json)

            # normalization range
            new_range = (0, 1)

            # calls callback for every frame
            for frame_index in range(frame_count):
                raw_frame_annotations = annotations.get(frame_index, [])

                # normalize annotations
                normalized_annotations = [
                    (
                        NorsvinBehaviorClass.from_json_label(behavior),
                        *AnnotationNormalizer.normalize_bounding_box(
                            image_dimensions=(original_width, original_height),
                            bounding_box=(x, y, w, h),
                            new_range=new_range,
                        )
                    )
                    for behavior, x, y, w, h in raw_frame_annotations
                ]
                print(f"[AnnotationLoader] Processed annotation for frame {frame_index} for source {source}")
                print(f"[AnnotationLoader] Annotation {frame_index}: {normalized_annotations}")
                self.callback(source, frame_index, normalized_annotations, False)

            # send termination signal
            self.callback(source, None, None, True)
        except Exception as e:
            print(f"Error in _process_annotations: {e}")
