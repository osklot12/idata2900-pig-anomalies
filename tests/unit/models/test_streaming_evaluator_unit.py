from typing import List, Optional

import numpy as np

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataclasses.bbox import BBox
from src.data.dataclasses.source_metadata import SourceMetadata
from src.data.dataset.streams.providers.stream_provider import StreamProvider
from src.data.dataset.streams.stream import Stream
from src.models.prediction import Prediction
from src.models.predictor import Predictor
from src.models.streaming_evaluator import StreamingEvaluator
from tests.utils.dummy_annotation_label import DummyAnnotationLabel
from tests.utils.generators.dummy_frame_generator import DummyFrameGenerator


class DummyPredictor(Predictor):
    """Dummy predictor for testing purposes."""

    def predict(self, image: np.ndarray) -> List[Prediction]:
        return [
            Prediction(x1=20, y1=30, x2=60, y2=90, conf=0.8, cls=1),
            Prediction(x1=25, y1=30, x2=60, y2=90, conf=0.6, cls=0),
            Prediction(x1=70, y1=100, x2=100, y2=200, conf=0.2, cls=0)
        ]

class DummyStream(Stream[AnnotatedFrame]):
    """Dummy stream for testing purposes."""

    def __init__(self):
        self.streamed = False

    def read(self) -> Optional[AnnotatedFrame]:
        result = None

        if not self.streamed:
            result = AnnotatedFrame(
                source=SourceMetadata(source_id="test", frame_resolution=(640, 640)),
                index=0,
                frame=DummyFrameGenerator.generate(640, 640),
                annotations=[
                    AnnotatedBBox(
                        cls=DummyAnnotationLabel.DEBUGGING,
                        bbox=BBox(x=26, y=30, width=40, height=60),
                    ),
                    AnnotatedBBox(
                        cls=DummyAnnotationLabel.CODING,
                        bbox=BBox(x=65, y=100, width=40, height=70)
                    )
                ]
            )
            self.streamed = True

        return result

class DummyStreamProvider(StreamProvider[AnnotatedFrame]):
    """Dummy StreamProvider for testing purposes."""

    def get_stream(self) -> Stream[AnnotatedFrame]:
        return DummyStream()

def test_streaming_evaluator_produces_correct_confusion_matrix():
    """Tests that the StreamingEvaluator produces the expected confusion matrix."""
    # arrange
    evaluator = StreamingEvaluator(
        stream_provider=DummyStreamProvider(),
        classes=["BACKGROUND", "CODING", "DEBUGGING"],
        class_shift=1,
        iou_thresh=0.5
    )
    predictor = DummyPredictor()

    # act
    evaluator.evaluate(predictor)