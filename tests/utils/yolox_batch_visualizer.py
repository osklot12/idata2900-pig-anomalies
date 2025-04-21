from typing import List

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt


class YOLOXBatchVisualizer:
    """Visualizer for YOLOX batches."""

    @staticmethod
    def visualize(images: torch.Tensor, targets: torch.Tensor, class_names: List[str]):
        """
        Visualizes a YOLOX-style batch.

        Args:
            images (torch.Tensor): (B, 3, H, W) - normalized between 0 and 1
            targets (torch.Tensor): (B, max_boxes, 5) - [cls, cx, cy, w, h]
            class_names (List[str]): list of class names
        """
        batch_size = images.shape[0]

        for i in range(batch_size):
            img = images[i].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            img = np.ascontiguousarray(img)

            height, width, _ = img.shape
            boxes = targets[i]

            for box in boxes:
                cls, cx, cy, w, h = box.tolist()
                print(f"Box: class={cls}, cx={cx}, cy={cy}, w={w}, h={h}")
                if cls >= 0:
                    cls = int(cls)
                    x1 = int(cx - w / 2)
                    y1 = int(cy - h / 2)
                    x2 = int(cx + w / 2)
                    y2 = int(cy + h / 2)

                    label = class_names[cls] if cls < len(class_names) else f"class_{cls}"

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 0), 1, cv2.LINE_AA)

                plt.figure(figsize=(6, 6))
                plt.imshow(img)
                plt.title(f"YOLOX Ground Truth (Image {i})")
                plt.axis("off")
                plt.show()