# src/utils/batch_input_visualizer.py
import os
import cv2
import numpy as np
import torch

def visualize_batch_input(images: torch.Tensor, bboxes: torch.Tensor, cls: torch.Tensor,
                          batch_idx: torch.Tensor, save_dir: str, prefix: str = "input"):
    os.makedirs(save_dir, exist_ok=True)
    B, _, H, W = images.shape

    for i in range(B):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = np.ascontiguousarray(img)

        boxes = bboxes[batch_idx == i]
        labels = cls[batch_idx == i]

        for box, label in zip(boxes, labels):
            x_center, y_center, w, h = box
            x1 = int((x_center - w / 2))
            y1 = int((y_center - h / 2))
            x2 = int((x_center + w / 2))
            y2 = int((y_center + h / 2))

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{int(label)}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)

        save_path = os.path.join(save_dir, f"{prefix}_frame_{i}.jpg")
        cv2.imwrite(save_path, img)
        print(f"🖼 Saved: {save_path}")
