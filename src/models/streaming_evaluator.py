import os
from typing import List, Optional
import datetime

import numpy as np
from rich.table import Table
from scipy.optimize import linear_sum_assignment
from torch.utils.tensorboard import SummaryWriter

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset.streams.providers.stream_provider import StreamProvider
from src.models.evalutor_visualizer import EvaluatorVisualizer
from src.models.prediction import Prediction
from src.models.predictor import Predictor
from src.utils.calculators.metrics.confusion_calculator import ConfusionCalculator
from src.utils.calculators.metrics.f1_calculator import F1Calculator
from src.utils.calculators.metrics.iou_calculator import IoUCalculator
from src.utils.calculators.metrics.map_calculator import MAPCalculator
from src.utils.logging import console

BATCH_SIZE = 300


class StreamingEvaluator:
    """Computes evaluation metrics with streaming compatibility."""

    def __init__(self, stream_provider: StreamProvider[AnnotatedFrame], classes: List[str], iou_thresh: float = 0.5,
                 output_dir: str = "faster_rcnn_outputs"):
        """
        Initializes a StreamingEvaluator instance.

        Args:
            stream_provider (StreamProvider[AnnotatedFrame]): provider of evaluation stream
            classes (List[str]): the class names in order
            iou_thresh (float): the iou threshold for predictions
            output_dir (str): output directory
        """
        self._stream_provider = stream_provider
        self._iou_thresh = iou_thresh
        self._classes: List[str] = classes
        self._map_calculator = MAPCalculator(num_classes=len(self._classes), iou_threshold=self._iou_thresh)
        self._output_dir = output_dir
        self._summary_writer = SummaryWriter(log_dir=f"{self._output_dir}/tensorboard")

    def evaluate(self, predictor: Predictor, epoch: Optional[int] = None) -> None:
        """
        Computes evaluation metrics for the predictor.

        Args:
            predictor (Predictor): the predictor to evaluate
            epoch (Optional[int]): optional epoch number
        """
        n_classes = len(self._classes)
        conf_mat = np.zeros((n_classes + 1, n_classes + 1))
        stream = self._stream_provider.get_stream()

        img_idx = 0
        while instance := stream.read():
            predictions = predictor.predict(instance.frame)
            gts = instance.annotations
            matches = self._get_gt_for_pred(predictions, gts)

            # compute map
            pred_np = np.array([[p.x1, p.y1, p.x2, p.y2, p.cls, p.conf] for p in predictions])
            gts_np = np.array([
                [g.bbox.x, g.bbox.y, g.bbox.x + g.bbox.width, g.bbox.y + g.bbox.height, g.cls.value]
                for g in instance.annotations
            ], dtype=np.float32)

            self._map_calculator.update(pred_np, gts_np)

            # compute confusion matrix
            pred_cls = [pred.cls for pred in predictions]
            gt_cls = [match.cls.value if match is not None else n_classes for match in matches]

            conf_mat += ConfusionCalculator.calculate(pred_cls, gt_cls, n_classes)

            # save image
            self._save_image(image=instance.frame, predictions=predictions, gts=gts, image_idx=img_idx, epoch=epoch)

            img_idx += 1

        self._write_confusion_matrix(conf_mat)
        self._write_f1(conf_mat)

        map_result = self._map_calculator.compute()
        self._write_map(map_result["mAP"], map_result["per_class_ap"])

    def _save_image(self, image: np.ndarray, predictions: List[Prediction], gts: List[AnnotatedBBox],
                     image_idx: int, epoch: Optional[int] = None) -> None:
        """Saves visualizations of predictions and ground truths on the images."""
        base_dir = os.path.join(self._output_dir, "eval_images")

        if epoch is not None:
            output_dir = os.path.join(base_dir, f"epoch_{epoch}")
        else:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = os.path.join(base_dir, f"run_{timestamp}")

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"frame_{image_idx}.jpg")

        EvaluatorVisualizer.save_image(
            image=image,
            predictions=predictions,
            ground_truths=gts,
            class_names=self._classes,
            save_path=save_path
        )

    def _write_confusion_matrix(self, matrix: np.ndarray, background_label: str = "background") -> None:
        """Prints a confusion matrix to the console."""
        n = len(self._classes)
        names = self._classes + [background_label]

        table = Table(title="Confusion Matrix")
        table.add_column("GT \\ Pred", justify="right", style="bold")
        for name in names:
            table.add_column(name, justify="right")

        for i, row in enumerate(matrix):
            table.add_row(names[i], *[str(int(cell)) for cell in row])

        console.print(table)

    def _write_f1(self, matrix: np.ndarray, epoch: Optional[int] = None) -> None:
        """Prints the F1 scores to the console."""
        f1_scores = [F1Calculator.calculate(matrix, i) for i in range(len(self._classes))]

        table = Table(title="F1 Scores")
        table.add_column("Class", justify="right", style="bold")
        table.add_column("F1 Score", justify="right", style="bold")

        for cls_name, f1 in zip(self._classes, f1_scores):
            table.add_row(cls_name, f"{f1:.4f}")

        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        table.add_row("[bold yellow]Macro Avg[/bold yellow]", f"[bold yellow]{macro_f1:.4f}[/bold yellow]")

        console.print(table)

        if self._summary_writer:
            epoch = epoch if epoch is not None else 0
            for label, score in zip(self._classes, f1_scores):
                self._summary_writer.add_scalar(f"eval/f1_{label}", score, epoch)

            self._summary_writer.add_scalar("eval/macro_f1", macro_f1, epoch)

    def _write_map(self, map_score: float, per_class_ap: dict, epoch: Optional[int] = None) -> None:
        """Prints mAP and per-class AP using rich table."""
        table = Table(title="AP Scores")
        table.add_column("Class", justify="right", style="bold")
        table.add_column("AP", justify="right", style="bold")

        for cls_id, ap in sorted(per_class_ap.items()):
            class_name = self._classes[cls_id] if cls_id < len(self._classes) else f"Class {cls_id}"
            table.add_row(class_name, f"{ap:.4f}")

        table.add_row("[bold yellow]mAP[/bold yellow]", f"[bold yellow]{map_score:.4f}[/bold yellow]")
        console.print(table)

        if self._summary_writer:
            epoch = epoch if epoch is not None else 0
            for cls_id, ap in per_class_ap.items():
                label = self._classes[cls_id] if cls_id < len(self._classes) else f"class_{cls_id}"
                self._summary_writer.add_scalar(f"eval/ap_{label}", ap, epoch)
            self._summary_writer.add_scalar("eval/mAP", map_score, epoch)

    def _get_gt_for_pred(self, predictions: List[Prediction], gts: List[AnnotatedBBox]) -> List[
        Optional[AnnotatedBBox]]:
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
