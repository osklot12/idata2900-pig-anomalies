from typing import List, Optional

import numpy as np
from rich.table import Table
from scipy.optimize import linear_sum_assignment

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset.streams.stream import Stream
from src.models.prediction import Prediction
from src.models.predictor import Predictor
from src.utils.calculators.metrics.confusion_calculator import ConfusionCalculator
from src.utils.calculators.metrics.iou_calculator import IoUCalculator
from src.utils.logging import console

BATCH_SIZE = 300


class StreamingEvaluator:
    """Computes evaluation metrics with streaming compatibility."""

    def __init__(self, eval_stream: Stream[AnnotatedFrame], classes: List[str], iou_thresh: float = 0.5):
        """
        Initializes a StreamingEvaluator instance.

        Args:
            eval_stream (Stream): the evaluation stream
            iou_thresh (float): the iou threshold for predictions
        """
        self._eval_stream = eval_stream
        self._iou_thresh = iou_thresh
        self._classes: List[str] = classes


    def evaluate(self, predictor: Predictor):
        """
        Computes evaluation metrics for the predictor.

        Args:
            predictor (Predictor): the predictor to evaluate
        """
        n_classes = len(self._classes)
        conf_mat = np.zeros((n_classes, n_classes))
        while instance := self._eval_stream.read():
            predictions = predictor.predict(instance.frame)
            matches = self._get_gt_for_pred(predictions, instance.annotations)

            pred_cls = [pred.cls for pred in predictions]
            gt_cls = [match.cls if match is not None else n_classes for match in matches]

            conf_mat += ConfusionCalculator.calculate(pred_cls, gt_cls, n_classes)
            
        self._print_confusion_matrix(conf_mat)

    def _print_confusion_matrix(self, matrix: np.ndarray, background_label: str = "background"):
        """Prints a confusion matrix to the console."""
        n = len(self._classes)
        names = self._classes + [background_label]

        table = Table(title="Confusion matrix")
        table.add_column("GT \ Pred", justify="right", style="bold")
        for name in names:
            table.add_column(name, justify="right")

        for i, row in enumerate(matrix):
            table.add_row(names[i], *[str(cell) for cell in row])

        console.print(table)

    def _get_gt_for_pred(self, predictions: List[Prediction], gts: List[AnnotatedBBox]) -> List[Optional[AnnotatedBBox]]:
        """Returns the ground truth that has the largest IoU with the given prediction."""
        pred_boxes = [
            [pred.x1, pred.y1, pred.x2, pred.y2] for pred in predictions
        ]
        gt_boxes = [
            [gt.bbox.x, gt.bbox.y, gt.bbox.x + gt.bbox.width, gt.bbox.y + gt.bbox.height] for gt in gts
        ]

        gt_matches = [None] * len(predictions)
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            iou_matrix = IoUCalculator.calculate(pred_boxes=np.array(pred_boxes), gt_boxes=np.array(gt_boxes))
            cost_matrix = 1.0 - iou_matrix

            pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

            for pi, gi in zip(pred_indices, gt_indices):
                if iou_matrix[pi, gi] >= self._iou_thresh:
                    gt_matches[pi] = gts[gi]

        return gt_matches