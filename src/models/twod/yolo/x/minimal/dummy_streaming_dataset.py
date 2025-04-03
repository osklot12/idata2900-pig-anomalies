import torch
from torch.utils.data import IterableDataset
import numpy as np

class DummyStreamingDataset(IterableDataset):
    def __init__(self, num_batches=100):
        self.num_batches = num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            img = np.random.rand(3, 640, 640).astype(np.float32)  # CHW
            boxes = np.array([[100, 100, 200, 200, 0]], dtype=np.float32)  # x1, y1, x2, y2, class
            yield {
                "img": torch.from_numpy(img),
                "gt_bboxes": torch.from_numpy(boxes[:, :4]),
                "gt_classes": torch.from_numpy(boxes[:, 4]).long(),
                "gt_scores": torch.ones((1,), dtype=torch.float32),  # Optional
                "img_info": torch.tensor([640, 640]),  # h, w
                "img_id": torch.tensor([0]),
            }