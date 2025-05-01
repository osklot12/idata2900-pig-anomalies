# src/models/validators/streaming_obb_validator.py

from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.utils.metrics import ConfusionMatrix, box_iou
import torch


class StreamingOBBValidator(OBBValidator):
    def __init__(self, dataloader, class_names, writer=None):
        super().__init__()
        self.dataloader = dataloader
        self.names = class_names
        self.nc = len(class_names)
        self.writer = writer
        self.device = None
        self.model = None

    def __call__(self, trainer=None, model=None):
        print("🧪 Custom validator triggered!")

        # This is how Ultralytics calls the validator
        if trainer:
            self.device = trainer.device
            self.model = trainer.model
            self.epoch = getattr(trainer, "epoch", 0)
        elif model:
            self.device = next(model.parameters()).device
            self.model = model
            self.epoch = 0
        else:
            raise ValueError("StreamingOBBValidator needs either trainer or model.")

        self.training = False
        self.metrics = {}
        self.stats = []
        self.confusion_matrix = ConfusionMatrix(nc=self.nc)

        for i, batch in enumerate(self.dataloader):
            print(f"🔍 Eval batch {i+1}/{len(self.dataloader)}")
            with torch.no_grad():
                imgs = batch["img"].to(self.device, non_blocking=True)
                preds = self.model(imgs)

                targets = {
                    "cls": batch["instances"]["cls"].to(self.device),
                    "bboxes": batch["instances"]["bboxes"].to(self.device),
                    "batch_idx": batch["batch_idx"].to(self.device),
                }

                self.update_metrics(preds, imgs, targets)

        self.metrics = self.get_stats()
        print("✅ Done evaluating. Metrics:", self.metrics)

        if self.writer:
            for key, value in self.metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"eval/{key}", value, self.epoch)
            self.writer.flush()

        return self.metrics

    def update_metrics(self, preds, imgs, targets):
        for i in range(len(imgs)):
            pred_i = preds[i]
            gt_mask = targets["batch_idx"] == i

            gt_cls = targets["cls"][gt_mask]
            gt_boxes = targets["bboxes"][gt_mask]

            if gt_boxes.numel() == 0 or pred_i.numel() == 0:
                print(f"⚠️ Skipping empty batch {i}")
                continue

            pred_boxes = pred_i[:, :5]
            pred_scores = pred_i[:, 5]
            pred_cls = pred_i[:, 6].long()

            ious = box_iou(pred_boxes[:, :4], gt_boxes[:, :4])
            self.stats.append((pred_scores.cpu(), pred_cls.cpu(), gt_cls.cpu(), ious.cpu()))
