from typing import Dict, List, Any

import numpy as np
import torch.nn
from tqdm import tqdm

from yolox.data import DataLoader
from yolox.evaluators.voc_eval import voc_ap
from yolox.utils import postprocess

EPSILON = 1e-6

POST_PROCESS_CONF_THRE = 0.1
POST_PROCESS_NMS_THRE = 0.65


class StreamingEvaluator:
    """A custom evaluator for YOLOX models that supports streaming evaluation."""

    def __init__(self, model: torch.nn.Module, dataloader: DataLoader, device: torch.device, num_classes: int,
                 iou_thresh: float = 0.5):
        """
        Initializes a StreamingEvaluator instance.

        Args:
            model (torch.nn.Module): the YOLOX model to evaluate
            dataloader (DataLoader): a dataloader yielding batches with 'images' and 'targets'
            device (torch.device): the device to run inference on (CPU or CUDA)
            num_classes (int): number of object classes
            iou_thresh (float): iou threshold to count a prediction as a true positive
        """
        self._model = model
        self._dataloader = dataloader
        self._device = device
        self._num_classes = num_classes
        self._iou_thresh = iou_thresh

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluates the model over the dataset.

        Returns:
            Dict[str, float]: a dictionary containing precision, recall, F1 score, and mAP
        """
        self._model.eval()
        all_detections: List[np.ndarray] = []
        all_annotations: List[np.ndarray] = []

        for batch in tqdm(self._dataloader, desc="Evaluating"):
            images, targets, _, _ = batch
            images = images.to(self._device)

            with torch.no_grad():
                outputs = self._model(images)

            all_detections.extend(self._convert_outputs(outputs))
            all_annotations.extend(self._convert_targets(targets))

        print("[Evaluator] Finished all batch processing.")

        metrics = self._compute_metrics(all_detections, all_annotations)

        print("[Evaluator] Returning metrics...")
        return metrics

    def _convert_outputs(self, outputs: torch.Tensor) -> List[np.ndarray]:
        """
        Converts model predictions to NumPy arrays.

        Args:
            outputs (torch.Tensor): model outputs

        Returns:
            List[np.ndarray]: list of detections per sample
        """
        detections = []

        processed_outputs = postprocess(
            outputs, self._num_classes, conf_thre=POST_PROCESS_CONF_THRE, nms_thre=POST_PROCESS_NMS_THRE
        )

        for preds in processed_outputs:
            if preds is None or preds.shape[0] == 0:
                detections.append(np.zeros((0, 6), dtype=np.float32))

            else:
                preds_np = preds.cpu().numpy().astype(np.float32)
                detections.append(preds_np)
            
        return detections

    @staticmethod
    def _convert_targets(targets: List[torch.Tensor]) -> List[np.ndarray]:
        """
        Converts ground truth targets to NumPy arrays.

        Args:
            targets (List[torch.Tensor]): list of ground truth annotations

        Returns:
            List[np.ndarray]: list of annotations per sample
        """
        annotations = []
        for image_targets in targets:
            valid = image_targets[:, 0] >= 0
            boxes = image_targets[valid]

            converted = []
            for box in boxes:
                cls, cx, cy, w, h = box.tolist()
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                converted.append([x1, y1, x2, y2, cls])

            annotations.append(np.array(converted, dtype=np.float32))

        return annotations

    def _compute_metrics(self, detections: List[np.ndarray], annotations: List[np.ndarray]) -> Dict[str, float]:
        """
        Computes detection metrics based on predictions and ground truth.

        Args:
            detections (List[np.ndarray]): list of predicted bounding boxes per image
            annotations (List[np.ndarray]): list of ground truth bounding boxes per image

        Returns:
            Dict[str, float]: computed metrics: precision, recall, f1 score, map
        """
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        labels = np.zeros((0,))
        conf_matrix = np.zeros((self._num_classes + 1, self._num_classes), dtype=int)
        n_gt = 0

        # iterate through each image
        for i, (pred, gt) in enumerate(zip(detections, annotations)):
            print(f"[Evaluator] Pair {i}: {len(pred)} preds, {len(gt)} GTs")
            detected_gt_indices = set()

            # iterate over each prediction
            for pred_box in pred:
                # predicted class
                pred_class = int(pred_box[5])
                # append confidence
                scores = np.append(scores, pred_box[4])
                # append predicated class
                labels = np.append(labels, pred_class)

                # if no gt is present, any prediction is a false positive
                if gt.size == 0:
                    true_positives = np.append(true_positives, 0)
                    conf_matrix[self._num_classes, pred_class] += 1

                else:
                    # get the gt that gives the highest iou for the prediction
                    ious = self._compute_iou(gt[:, :4], pred_box[:4])
                    max_iou_idx = int(np.argmax(ious))
                    max_iou = ious[max_iou_idx]
                    gt_class = int(gt[max_iou_idx][4])

                    # iou meets threshold
                    if max_iou >= self._iou_thresh:
                        # correct prediction!
                        if max_iou_idx not in detected_gt_indices and pred_class == gt_class:
                            true_positives = np.append(true_positives, 1)
                            conf_matrix[gt_class, pred_class] += 1
                            detected_gt_indices.add(max_iou_idx)

                        # either wrong class or this gt box already matched
                        else:
                            true_positives = np.append(true_positives, 0)
                            conf_matrix[gt_class, pred_class] += 1

                    # bounding box is off
                    else:
                        true_positives = np.append(true_positives, 0)
                        # false positive from background
                        conf_matrix[self._num_classes, pred_class] += 1

            n_gt += gt.shape[0]

        print("[Evaluator] Finished loop over detections and annotations.")
        print(f"[Evaluator] #TP: {len(true_positives)}, #Scores: {len(scores)}, #Labels: {len(labels)}, GT: {n_gt}")
        print("[Evaluator] Calling _calculate_scores()...")

        precision, recall, f1, ap, ap_dict = self._calculate_scores(true_positives, scores, labels, n_gt)

        print("[Evaluator] Done computing scores.")
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mAP": ap,
            "per_class_ap": ap_dict,
            "confusion_matrix": conf_matrix
        }

    def _compute_iou(self, boxes1: np.ndarray, box2: np.ndarray) -> np.ndarray:
        """
        Computes IoU between multiple ground truth boxes and a single prediction.

        Args:
            boxes1 (np.ndarray): ground truth boxes (N, 4).
            box2 (np.ndarray): predicted box (4,).

        Returns:
            np.ndarray: iou values (N,).
        """
        inter_x1 = np.maximum(boxes1[:, 0], box2[0])
        inter_y1 = np.maximum(boxes1[:, 1], box2[1])
        inter_x2 = np.minimum(boxes1[:, 2], box2[2])
        inter_y2 = np.minimum(boxes1[:, 3], box2[3])

        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)

        box1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        return inter_area / np.maximum(union_area, 1e-6)

    def _calculate_scores(self, tp: np.ndarray, scores: np.ndarray, labels: np.ndarray, n_gt: int
                          ) -> (float, float, float, float, Dict[int, float]):
        """
        Calculates precision, recall, F1, and mAP from evaluation results.

        Args:
            tp (np.ndarray): true positives
            scores (np.ndarray): confidence scores
            labels (np.ndarray): class labels
            n_gt (int): total number of ground truth boxes

        Returns:
            Tuple[float, float, float, float]: precision, recall, f1 score, map
        """
        indices = np.argsort(-scores)
        tp = tp[indices]
        labels = labels[indices]

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(1 - tp)

        recall = cum_tp / (n_gt + EPSILON)
        precision = cum_tp / (cum_tp + cum_fp + EPSILON)
        f1 = 2 * precision * recall / (precision + recall + EPSILON)

        ap, ap_dict = self._mean_average_precision(tp, labels)

        return float(precision[-1]), float(recall[-1]), float(f1[-1]), ap, ap_dict

    def _mean_average_precision(self, tp: np.ndarray, labels: np.ndarray) -> (float, Dict[int, float]):
        """
        Computes mean Average Precision (mAP) across all classes and per-class AP.

        Args:
            tp (np.ndarray): True positives sorted by confidence.
            labels (np.ndarray): Class labels corresponding to predictions.

        Returns:
            Tuple[float, Dict[int, float]]: map and per-class ap
        """
        ap_dict = {}
        for c in range(self._num_classes):
            idx = labels == c
            if np.sum(idx):
                class_tp = tp[idx]
                cum_tp = np.cumsum(class_tp)
                cum_fp = np.cumsum(1 - class_tp)
                recall = cum_tp / (np.sum(class_tp) + EPSILON)
                precision = cum_tp / (cum_tp + cum_fp + EPSILON)
                ap_dict[c] = voc_ap(recall, precision)
            else:
                ap_dict[c] = 0.0

        mean_ap = float(np.mean(list(ap_dict.values())))
        return mean_ap, ap_dict
