from typing import List, Dict
import numpy as np
import torch

from src.data.dataclasses.annotated_frame import AnnotatedFrame


class UltralyticsBatchConverter:
    """Converts a batch into Ultralytics OBB training format."""

    @staticmethod
    def convert(batch: List[AnnotatedFrame]) -> List[dict]:
        """
        Converts a list of AnnotatedFrame objects into the dictionary format expected by Ultralytics.

        Each sample will include:
        - img: torch.Tensor of shape (3, H, W)
        - instances: dict with 'cls' and 'bboxes'
        - batch_idx, im_file, ori_shape, ratio_pad
        """
        results = []
        for i, frame in enumerate(batch):
            img = torch.from_numpy(frame.frame).permute(2, 0, 1).float() / 255.0  # HWC -> CHW
            cls = torch.tensor([ann.cls.value for ann in frame.annotations], dtype=torch.long)
            bboxes = torch.tensor([
                [
                    (ann.bbox.x + ann.bbox.width / 2) / frame.frame.shape[1],
                    (ann.bbox.y + ann.bbox.height / 2) / frame.frame.shape[0],
                    ann.bbox.width / frame.frame.shape[1],
                    ann.bbox.height / frame.frame.shape[0],
                    0.0  # angle = 0.0 for now
                ]
                for ann in frame.annotations
            ], dtype=torch.float32)

            if len(cls) == 0:
                cls = torch.empty((0,), dtype=torch.long)
                bboxes = torch.empty((0, 5), dtype=torch.float32)

            results.append({
                "img": img,
                "instances": {
                    "cls": cls,
                    "bboxes": bboxes,
                },
                "batch_idx": torch.full((len(cls),), i, dtype=torch.long),
                "im_file": [f"frame_{i}.jpg"],
                "ori_shape": [torch.tensor([frame.frame.shape[0], frame.frame.shape[1]])],
                "ratio_pad": [(torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0]))],
            })

        return results

