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
        self.writer = writer  # TensorBoard SummaryWriter

    def __call__(self, trainer):
        print("🧪 Starting custom streaming evaluation...")
        self.device = trainer.device
        self.model = trainer.model
        self.training = False
        self.metrics = {}
        self.stats = []
        self.confusion_matrix = ConfusionMatrix(nc=self.nc)
        self.pbar = enumerate(self.dataloader)


        for i, batch in enumerate(self.dataloader):
            print(f"🔍 Evaluating batch {i + 1}/{len(self.dataloader)}")
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
        print("✅ Evaluation done. Metrics:", self.metrics)

        # 📈 Optional TensorBoard logging
        if self.writer:
            epoch = trainer.epoch
            for key, value in self.metrics.items():
                if isinstance(value, (float, int)):
                    self.writer.add_scalar(f"eval/{key}", value, epoch)
            self.writer.flush()

        return self.metrics

    def update_metrics(self, preds, imgs, targets):
        for i in range(len(imgs)):
            pred_i = preds[i]
            gt_mask = targets["batch_idx"] == i

            gt_cls = targets["cls"][gt_mask]
            gt_boxes = targets["bboxes"][gt_mask]

            if gt_boxes.numel() == 0 or pred_i.numel() == 0:
                print(f"⚠️ Skipping batch {i} — empty predictions or targets")
                continue

            pred_boxes = pred_i[:, :5]
            pred_scores = pred_i[:, 5]
            pred_cls = pred_i[:, 6].long()

            ious = box_iou(pred_boxes[:, :4], gt_boxes[:, :4])
            self.stats.append((pred_scores.cpu(), pred_cls.cpu(), gt_cls.cpu(), ious.cpu()))
