from typing import List
import os
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt


class YOLOv8BatchVisualizer:
    """Visualizer for YOLOv8 batches with GT and predictions."""

    @staticmethod
    def visualize(images: torch.Tensor, targets: List[np.ndarray], class_names: List[str]):
        """
        Visualizes a YOLOv8-style batch.

        Args:
            images (torch.Tensor): (B, 3, H, W) - normalized 0–1
            targets (List[np.ndarray]): list of (N, 5) numpy arrays [cls, x1, y1, x2, y2]
            class_names (List[str]): list of class names
        """
        for i, (image_tensor, target) in enumerate(zip(images, targets)):
            img = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            for box in target:
                cls, x1, y1, x2, y2 = box.astype(int)
                label = class_names[cls] if cls < len(class_names) else f"class_{cls}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)

            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.title(f"YOLOv8 Ground Truth (Image {i})")
            plt.axis("off")
            plt.show()

    @staticmethod
    def visualize_with_predictions(images: torch.Tensor, targets: List[np.ndarray],
                                   predictions: List[np.ndarray], class_names: List[str],
                                   start_idx: int = 0, save_dir: str = "./eval_visuals"):
        """
        Saves visualizations with both ground truth and predicted boxes to image files.

        Args:
            images (torch.Tensor): (B, 3, H, W)
            targets (List[np.ndarray]): each entry is (N, 5) [cls, x1, y1, x2, y2]
            predictions (List[np.ndarray]): each entry is (N, 6) [x1, y1, x2, y2, conf, cls]
            class_names (List[str]): list of class names
            start_idx (int): starting image index
            save_dir (str): directory to save results
        """
        os.makedirs(save_dir, exist_ok=True)

        for i, (image_tensor, target_boxes, pred_boxes) in enumerate(zip(images, targets, predictions)):
            img = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            # Draw GT boxes
            for box in target_boxes:
                cls, x1, y1, x2, y2 = box.astype(int)
                label = class_names[cls] if cls < len(class_names) else f"class_{cls}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"GT: {label}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # Draw predicted boxes
            for pred in pred_boxes:
                if len(pred) < 6:
                    continue  # skip malformed predictions
                x1, y1, x2, y2, conf, cls = pred[:6]
                cls = int(cls)
                label = class_names[cls] if cls < len(class_names) else f"class_{cls}"
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(img, f"P: {label} ({conf:.2f})", (int(x1), int(y2) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            save_path = os.path.join(save_dir, f"eval_image_{start_idx + i}.jpg")
            cv2.imwrite(save_path, img)
            print(f"Saved visualization to: {save_path}")
