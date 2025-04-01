import os
import torch
import logging
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from src.data.streaming.prefetchers.batch_prefetcher import BatchPrefetcher
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
    """
    RCNN Trainer
    """

    def __init__(self, prefetcher: BatchPrefetcher):
        self.prefetcher = prefetcher
        self.model = None

    def setup(self):
        self.model = self.create_model()

    def create_model(self):
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
        return model.cuda()

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

            # Send metrics every iteration
            predictions = model(images)
            schema = SignedSchema(
                issuer_id=issuer_id,
                schema=formatter.format(predictions, targets)
            )
            metrics_broker.notify(schema)

            # Save checkpoint every 100 iterations
            if iteration % 100 == 0 and iteration > 0:
                checkpoint_path = os.path.join("checkpoints", f"model_checkpoint_{iteration}.pt")
                torch.save(model.state_dict(), checkpoint_path)

        # Final model save
        torch.save(model.state_dict(), os.path.join("checkpoints", "model_final.pt"))
        return "Training completed."

    def evaluate(self) -> str:
        return "Evaluation not implemented yet."

    def _convert_to_tensors(self, batch):
        images = []
        targets = []
        for frame in batch:
            images.append(torch.tensor(frame.image, dtype=torch.float32, device="cuda").permute(2, 0, 1) / 255.0)
            boxes = torch.tensor([
                [ann.bbox[0], ann.bbox[1], ann.bbox[0] + ann.bbox[2], ann.bbox[1] + ann.bbox[3]]
                for ann in frame.annotations
            ], dtype=torch.float32, device="cuda")
            labels = torch.tensor([ann.category_id for ann in frame.annotations], dtype=torch.int64, device="cuda")
            targets.append({"boxes": boxes, "labels": labels})
        return images, targets
