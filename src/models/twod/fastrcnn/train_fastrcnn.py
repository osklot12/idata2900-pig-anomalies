import os
import time

import torch
import logging
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from src.data.dataset.streams.prefetcher import Prefetcher
from src.models.model_trainer import ModelTrainer

from src.schemas.metric_schema import MetricSchema
from src.schemas.observer.schema_broker import SchemaBroker
from src.schemas.signed_schema import SignedSchema
from src.schemas.training_metric_formater import TrainingMetricsFormatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

metrics_broker = SchemaBroker[SignedSchema[MetricSchema]]()
formatter = TrainingMetricsFormatter()
issuer_id = "rcnn-trainer"


class RCNNTrainer(ModelTrainer):
    def __init__(self, prefetcher: Prefetcher):
        self.prefetcher = prefetcher
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(self):
        self.model = self.create_model()

    def create_model(self):
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
        return model.to(self.device)

    def train(self) -> str:
        self.setup()
        model = self.model
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        os.makedirs("checkpoints", exist_ok=True)

        for iteration in range(1000):
            batch = self.prefetcher.get()
            images, targets = self._convert_to_tensors(batch)

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            total_loss = sum(loss_dict.values())
            total_loss.backward()
            optimizer.step()

            predictions = model(images)
            schema = SignedSchema(
                issuer_id=issuer_id,
                schema=formatter.format(predictions, targets),
                timestamp=time.time()
            )
            metrics_broker.notify(schema)

            if iteration % 100 == 0 and iteration > 0:
                checkpoint_path = os.path.join("checkpoints", f"model_checkpoint_{iteration}.pt")
                torch.save(model.state_dict(), checkpoint_path)

        torch.save(model.state_dict(), os.path.join("checkpoints", "model_final.pt"))
        return "Training completed."

    def evaluate(self) -> str:
        return "Evaluation not implemented yet."

    def _convert_to_tensors(self, batch):
        images = []
        targets = []
        for frame in batch:
            images.append(
                torch.tensor(frame.frame, dtype=torch.float32, device=self.device).permute(2, 0, 1) / 255.0
            )
            boxes = torch.tensor([
                [ann.bbox[0], ann.bbox[1], ann.bbox[0] + ann.bbox[2], ann.bbox[1] + ann.bbox[3]]
                for ann in frame.annotations
            ], dtype=torch.float32, device=self.device)
            labels = torch.tensor(
                [ann.category_id for ann in frame.annotations],
                dtype=torch.int64,
                device=self.device
            )
            targets.append({"boxes": boxes, "labels": labels})
        return images, targets
