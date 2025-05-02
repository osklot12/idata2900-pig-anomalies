import pytest
from typing import List, Optional

import numpy as np

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset.streams.stream import Stream, T
from src.models.prediction import Prediction
from src.models.predictor import Predictor


class DummyPredictor(Predictor):
    """Dummy predictor for testing purposes."""

    def predict(self, image: np.ndarray) -> List[Prediction]:
        return [
            Prediction(x1=20, y1=30, x2=60, y2=90, conf=0.8, cls=0),
            Prediction(x1=25, y1=30, x2=60, y2=90, conf=0.6, cls=1),
            Prediction(x1=70, y1=100, x2=100, y2=200, conf=0.2, cls=2)
        ]

class DummyStream(Stream[AnnotatedFrame]):
    """Dummy stream for testing purposes."""

    def read(self) -> Optional[AnnotatedFrame]:
        return AnnotatedFrame(
            source=Meta
        )