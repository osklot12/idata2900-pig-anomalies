import threading
from typing import Callable, Dict, Tuple, Optional, List, Type

from src.data.annotation_enum_parser import AnnotationEnumParser
from src.data.decoders.bbox_decoder import BBoxDecoder
from src.data.loading.feed_status import FeedStatus
from src.data.bbox_normalizer import BBoxNormalizer
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from src.utils.source_normalizer import SourceNormalizer


class AnnotationLoader:
    """
    Loads annotations for video streams and invokes callable, feeding the annotations forward.
    """

    def __init__(self, data_loader, decoder_cls: Type[BBoxDecoder], label_parser: Type[AnnotationEnumParser],
                 callback: Callable[[str, int, Optional[List[Tuple[NorsvinBehaviorClass, float, float, float, float]]], bool], FeedStatus]):
        self.data_loader = data_loader
        self.decoder_cls = decoder_cls
        self.label_parser = label_parser
        self.callback = callback
        self._thread = None

    def load_annotations(self, annotation_blob_name: str):
        """
        Loads annotations for a video, feeding them to the callback function.

        Args:
            annotation_blob_name: blob name of the annotation file.
        """
        self._thread = threading.Thread(target=self._process_annotations, args=(annotation_blob_name,))
        self._thread.start()

    def wait_for_completion(self):
        """Waits for the annotation processing thread to finish."""
        if self._thread:
            self._thread.join()

    def _process_annotations(self, annotation_blob_name: str):
        """Processes annotations and feeds them to the callback function."""
        try:
            # download json annotation file
            annotations_json = self.data_loader.download_json(annotation_blob_name)

            # instantiate decoder
            decoder = self.decoder_cls(annotations_json)

            # fetch video name and normalize source id
            video_name = annotations_json.get("item", {}).get("name", "unknown_video")
            source = SourceNormalizer.normalize(video_name)

            # extract image dimensions (width, height)
            original_width, original_height = decoder.get_frame_dimensions()

            frame_count = decoder.get_frame_count()
            if frame_count <= 0:
                raise RuntimeError(f"No frames found in {annotation_blob_name}")

            raw_annotations: Dict[int, list] = decoder.get_annotations()

            # normalization range
            new_range = (0, 1)

           # normalize annotations
            normalized_annotations = {
                frame_index: BBoxNormalizer.normalize_annotations(
                    image_dimensions=(original_width, original_height),
                    annotations=annotations,
                    new_range=new_range,
                    annotation_parser=self.label_parser,
                )
                for frame_index, annotations in raw_annotations.items()
            }

            # calls callback for every frame
            for frame_index in range(frame_count):
                frame_annotations = normalized_annotations.get(frame_index, [])
                print(f"[AnnotationLoader] Processed annotation for frame {frame_index} for source {source}")
                print(f"[AnnotationLoader] Annotation {frame_index}: {frame_annotations}")
                self.callback(source, frame_index, frame_annotations, False)

            # send termination signal
            self.callback(source, None, None, True)
        except Exception as e:
            print(f"Error in _process_annotations: {e}")
