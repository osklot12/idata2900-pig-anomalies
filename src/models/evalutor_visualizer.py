import os
import cv2
from typing import List

import numpy as np

from src.models.prediction import Prediction
from src.data.dataclasses.annotated_bbox import AnnotatedBBox


class EvaluatorVisualizer:
    """Simple visualizer for evaluator outputs."""

    @staticmethod
    def save_image(
        image: np.ndarray,
        predictions: List[Prediction],
        ground_truths: List[AnnotatedBBox],
        class_names: List[str],
        save_path: str,
    ):
        img = image.copy()

        for gt in ground_truths:
            cls = int(gt.cls.value)
            label = class_names[cls] if cls < len(class_names) else f"class_{cls}"
            x1 = int(gt.bbox.x)
            y1 = int(gt.bbox.y)
            x2 = int(gt.bbox.x + gt.bbox.width)
            y2 = int(gt.bbox.y + gt.bbox.height)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"GT: {label}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)

        for p in predictions:
            cls = p.cls
            label = class_names[cls] if cls < len(class_names) else f"class_{cls}"
            conf = p.conf
            x1, y1, x2, y2 = map(int, [p.x1, p.y1, p.x2, p.y2])

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"P: {label} ({conf:.2f})", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)
