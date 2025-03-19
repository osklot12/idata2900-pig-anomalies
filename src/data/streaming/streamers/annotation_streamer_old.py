import threading
from typing import Callable, Dict, Tuple, Optional, List, Type

from src.data.decoders.annotation_decoder import AnnotationDecoder
from src.data.loading.feed_status import FeedStatus
from src.data.preprocessing.normalization.simple_bbox_normalizer import SimpleBBoxNormalizer
from src.data.streaming.streamers.streamer import Streamer
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from src.utils.source_normalizer import SourceNormalizer


class AnnotationStreamerOld(Streamer):
    """
    Loads annotations for video streams and invokes callable, feeding the annotations forward.
    """

    def __init__(self, data_loader, annotation_blob_name: str, decoder_cls: Type[AnnotationDecoder], normalizer: SimpleBBoxNormalizer,
                 callback: Callable[[str, int, Optional[List[Tuple[NorsvinBehaviorClass, float, float, float, float]]],
                                     bool], FeedStatus]):
        self.data_loader = data_loader
        self.annotation_blob_name = annotation_blob_name
        self.decoder_cls = decoder_cls
        self.normalizer = normalizer
        self.callback = callback
        self._thread = None

    def start_streaming(self):
        self._thread = threading.Thread(target=self._process_annotations, args=(self.annotation_blob_name,))
        self._thread.start()

    def stop(self):
        pass

    def wait_for_completion(self):
        if self._thread:
            self._thread.join()

    def _process_annotations(self, annotation_blob_name: str):
        """Processes annotations and feeds them to the callback function."""
        try:
            # download json annotation file
            json = self.data_loader.download_json(annotation_blob_name)

            # instantiate decoder
            decoder = self.decoder_cls(json)

            # extract and set image dimensions (width, height) for normalizer
            self.normalizer.set_image_dimensions(decoder.get_frame_dimensions())

            # extract metadata
            source = self._get_source_id(json)
            frame_count = self._get_frame_count(annotation_blob_name, decoder)

            # normalize annotations
            raw_annotations: Dict[int, list] = decoder.get_annotations()
            normalized_annotations = self._get_normalized_annotations(raw_annotations)

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

    def _get_normalized_annotations(self, raw_annotations):
        return {
            frame_index: self.normalizer.normalize_annotations(annotations)
            for frame_index, annotations in raw_annotations.items()
        }

    @staticmethod
    def _get_source_id(json):
        video_name = json.get("item", {}).get("name", "unknown_video")
        return SourceNormalizer.normalize(video_name)

    @staticmethod
    def _get_frame_count(annotation_blob_name, decoder):
        frame_count = decoder.get_frame_count()
        if frame_count <= 0:
            raise RuntimeError(f"No frames found in {annotation_blob_name}")
        return frame_count