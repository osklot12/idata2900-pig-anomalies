import os
import time

import torch
import logging

from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_iou

from src.data.streaming.prefetchers.batch_prefetcher import BatchPrefetcher
from src.models.model_trainer import ModelTrainer

from src.schemas.schemas.metric_schema import MetricSchema
from src.schemas.brokers.schema_broker import SchemaBroker
from src.schemas.schemas.signed_schema import SignedSchema
from src.schemas.training_metric_formater import TrainingMetricsFormatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

metrics_broker = SchemaBroker[SignedSchema[MetricSchema]]()
formatter = TrainingMetricsFormatter()
issuer_id = "rcnn-trainer"


class RCNNTrainer(ModelTrainer):
    def __init__(self, prefetcher: BatchPrefetcher):
        self._last_metrics = None
        self.prefetcher = prefetcher
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(self):
        self.model = self.create_model()

    def create_model(self):
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=4)
        return model.to(self.device)

    def train(self) -> str:
        self.setup()
        model = self.model
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        os.makedirs("checkpoints", exist_ok=True)

        for iteration in range(2):
            batch = self.prefetcher.get()
            images, targets = self._convert_to_tensors(batch)

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            total_loss = sum(loss_dict.values())
            total_loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                predictions = model(images)
            model.train()


            schema = SignedSchema(
                signature=issuer_id,
                schema=formatter.format(predictions, targets),
            )
            metrics_broker.notify(schema)

            if iteration % 100 == 0 and iteration > 0:
                checkpoint_path = os.path.join("checkpoints", f"model_checkpoint_{iteration}.pt")
                torch.save(model.state_dict(), checkpoint_path)

        torch.save(model.state_dict(), os.path.join("checkpoints", "model_final.pt"))
        return "Training completed."

    def evaluate(self, num_batches: int = 10) -> str:
        self.setup()
        self.model.eval()
        writer = SummaryWriter(log_dir="runs/rcnn_eval")  # ✅ logs to TensorBoard

        total_iou = 0.0
        total_predictions = 0
        total_gt_boxes = 0

        with torch.no_grad():
            for i in range(num_batches):
                batch = self.prefetcher.get()
                images, targets = self._convert_to_tensors(batch)
                predictions = self.model(images)

                for pred, target in zip(predictions, targets):
                    if pred["boxes"].numel() == 0 or target["boxes"].numel() == 0:
                        continue

                    ious = box_iou(pred["boxes"], target["boxes"])
                    max_ious = ious.max(dim=1).values
                    total_iou += max_ious.sum().item()
                    total_predictions += len(pred["boxes"])
                    total_gt_boxes += len(target["boxes"])

        avg_iou = total_iou / max(total_predictions, 1)
        recall = total_iou / max(total_gt_boxes, 1)

        # ✅ TensorBoard logs
        writer.add_scalar("eval/mAP50", avg_iou, 0)
        writer.add_scalar("eval/recall", recall, 0)
        writer.flush()
        writer.close()

        # ✅ Schema push
        schema = SignedSchema(
            signature=issuer_id,  # ✅ Use this instead of issuer_id=
            schema=MetricSchema(
                metrics={
                    "map50": avg_iou,
                    "recall": recall,
                    "precision": 0.0
                },
                timestamp=time.time()
            )
        )
        metrics_broker.notify(schema)

        self._last_metrics = {
            "map50": avg_iou,
            "recall": recall,
            "precision": 0.0
        }

        logger.info(f"✅ Evaluation done — IoU: {avg_iou:.3f}, Recall: {recall:.3f}")
        return "Evaluation completed."

    def _convert_to_tensors(self, batch):
        images = []
        targets = []
        for idx, frame in enumerate(batch):
            print(f"[Pre-Convert] Frame {idx} shape: {frame.frame.shape} — {len(frame.annotations)} annotations")

            images.append(
                torch.tensor(frame.frame, dtype=torch.float32, device=self.device).permute(2, 0, 1) / 255.0
            )

            if frame.annotations:
                for a_i, ann in enumerate(frame.annotations):
                    print(f" - Annotation {a_i}: class={ann.cls.value}, box={ann.bbox}")

                boxes = torch.tensor([[
                    ann.bbox.x * frame.frame.shape[1],
                    ann.bbox.y * frame.frame.shape[0],
                    (ann.bbox.x + ann.bbox.width) * frame.frame.shape[1],
                    (ann.bbox.y + ann.bbox.height) * frame.frame.shape[0]
                ] for ann in frame.annotations], dtype=torch.float32, device=self.device)

                labels = torch.tensor(
                    [ann.cls.value for ann in frame.annotations],
                    dtype=torch.int64,
                    device=self.device
                )
            else:
                boxes = torch.empty((0, 4), dtype=torch.float32, device=self.device)
                labels = torch.empty((0,), dtype=torch.int64, device=self.device)

            targets.append({"boxes": boxes, "labels": labels})
        return images, targets
