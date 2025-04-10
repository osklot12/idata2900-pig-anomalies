from typing import List
import torch
from src.data.dataclasses.annotated_frame import AnnotatedFrame


class UltralyticsBatchConverter:
    """Converts a batch into Ultralytics OBB training format."""

    @staticmethod
    def convert(batch: List[AnnotatedFrame]) -> List[dict]:
        """
        Converts a list of AnnotatedFrame objects into the dictionary format expected by Ultralytics.
        Assumes bounding boxes are already normalized.
        """
        results = []
        for i, frame in enumerate(batch):
            img = torch.from_numpy(frame.frame).permute(2, 0, 1).float() / 255.0  # HWC -> CHW

            # No extra normalization here, just pass the values directly
            cls_list = []
            bbox_list = []
            for ann in frame.annotations:
                cx = ann.bbox.x + ann.bbox.width / 2
                cy = ann.bbox.y + ann.bbox.height / 2
                bw = ann.bbox.width
                bh = ann.bbox.height

                cls_list.append(ann.cls.value)
                bbox_list.append([cx, cy, bw, bh, 0.0])  # angle = 0.0

                print(f"[Converter] class={ann.cls.value}, bbox=({cx}, {cy}, {bw}, {bh})")

            if len(cls_list) == 0:
                cls_tensor = torch.empty((0,), dtype=torch.long)
                bbox_tensor = torch.empty((0, 5), dtype=torch.float32)
            else:
                cls_tensor = torch.tensor(cls_list, dtype=torch.long)
                bbox_tensor = torch.tensor(bbox_list, dtype=torch.float32)

            results.append({
                "img": img,
                "instances": {
                    "cls": cls_tensor,
                    "bboxes": bbox_tensor,
                },
                "batch_idx": torch.full((len(cls_tensor),), i, dtype=torch.long),
                "im_file": [f"frame_{i}.jpg"],
                "ori_shape": [torch.tensor([frame.frame.shape[0], frame.frame.shape[1]])],
                "ratio_pad": [(torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0]))],
            })

            print(f"[Converter] Frame shape: {frame.frame.shape}, converted {len(cls_tensor)} bboxes")

        return results