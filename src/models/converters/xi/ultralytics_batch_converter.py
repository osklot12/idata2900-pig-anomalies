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

            cls_list = []
            bbox_list = []

            for ann in frame.annotations:
                cx = ann.bbox.x + ann.bbox.width / 2
                cy = ann.bbox.y + ann.bbox.height / 2
                bw = ann.bbox.width
                bh = ann.bbox.height

                # ✅ Filter out degenerate boxes
                if bw <= 0 or bh <= 0:
                    print(f"[Converter] ⚠️ Skipping invalid bbox: w={bw}, h={bh}")
                    continue

                cls_list.append(ann.cls.value)
                bbox_list.append([cx, cy, bw, bh, 0.0])
                print(f"[Converter] class={ann.cls.value}, bbox=({cx}, {cy}, {bw}, {bh})")

            # ✅ Clean malformed bboxes (e.g., [[]])
            bbox_list = [b for b in bbox_list if isinstance(b, list) and len(b) == 5]

            # 🛡 Safely convert cls_list to tensor
            if cls_list:
                cls_tensor = torch.tensor(cls_list, dtype=torch.long).reshape(-1)
            else:
                cls_tensor = torch.empty((0,), dtype=torch.long)

            # 🛡 Safely convert bbox_list to tensor
            if bbox_list:
                bbox_tensor = torch.tensor(bbox_list, dtype=torch.float32).reshape(-1, 5)
            else:
                bbox_tensor = torch.empty((0, 5), dtype=torch.float32)

            # 🧩 Match batch size for this sample
            batch_idx_tensor = torch.full((cls_tensor.shape[0],), i, dtype=torch.long)

            # ✅ Safety checks
            assert cls_tensor.ndim == 1, f"[Converter] Invalid cls_tensor shape: {cls_tensor.shape}"
            assert bbox_tensor.ndim == 2 and bbox_tensor.shape[1] == 5, f"[Converter] Invalid bbox_tensor shape: {bbox_tensor.shape}"

            results.append({
                "img": img,
                "instances": {
                    "cls": cls_tensor,
                    "bboxes": bbox_tensor,
                },
                "batch_idx": batch_idx_tensor,
                "im_file": [f"frame_{i}.jpg"],  # ✅ list of 1 string
                "ori_shape": [torch.tensor([frame.frame.shape[0], frame.frame.shape[1]])],  # ✅ list of 1 tensor
                "ratio_pad": [(torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0]))],  # ✅ list of 1 tuple of tensors
            })

            print(f"[Converter] Frame shape: {frame.frame.shape}, converted {len(cls_tensor)} bboxes")

        return results