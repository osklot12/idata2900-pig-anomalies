from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset.streams.stream import Stream
from src.models.predictor import Predictor


class StreamingEvaluator:
    """Computes evaluation metrics with streaming compatibility."""

    def __init__(self, predictor: Predictor, eval_stream: Stream[AnnotatedFrame], iou_thresh: float = 0.5):
        """
        Initializes a StreamingEvaluator instance.

        Args:
            predictor (Predictor): the predictor to use
            eval_stream (Stream): the evaluation stream
        """
        self._predictor = predictor
        self._eval_stream = eval_stream

    def evaluate(self):
        """
        Computes evaluation metrics for the predictor.
        """
        instance = self._eval_stream.read()
        while instance:
            pred = self._predictor.predict(instance.frame)
