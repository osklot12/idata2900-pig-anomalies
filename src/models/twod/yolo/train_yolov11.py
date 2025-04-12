import os
import time
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import v8DetectionLoss

from src.models.converters.batch_converter import BatchConverter
from src.models.converters.prediction_converter import PredictionConverter
from src.models.converters.target_converter import TargetConverter
from src.models.model_trainer import ModelTrainer
from src.models.twod.fastrcnn.train_fastrcnn import metrics_broker
from src.schemas.signed_schema import SignedSchema
from src.schemas.training_metric_formater import TrainingMetricsFormatter
from src.data.dataset.streams.prefetcher import Prefetcher

issuer_id = "yolov11"
formatter = TrainingMetricsFormatter()

class YOLOv11Trainer(ModelTrainer):
    def __init__(self, prefetcher: Prefetcher, model_variant: str = "yolo11m.pt", epochs: int = 50):
        self.prefetcher = prefetcher
        self.model_variant = model_variant
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo = YOLO(self.model_variant)
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.batch_converter = BatchConverter(self.device)
        self.pred_converter = PredictionConverter()
        self.target_converter = TargetConverter(self.device)

    def setup(self):
        self.model = self.yolo.model
        self.model.to(self.device)
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = v8DetectionLoss(self.model.head)

    def train(self) -> str:
        self.setup()
        os.makedirs("checkpoints", exist_ok=True)

        for iteration in range(self.epochs):
            batch = self.prefetcher.get()
            images, targets = self.batch_converter.convert_to_tuple_of_tensors(batch)

            self.optimizer.zero_grad()
            preds = self.model(images)

            # Compute YOLOv11 loss manually (NOTE: must be implemented or reused)
            yolo_targets = self.target_converter.convert_to_tensors(targets, images)

            loss, loss_items = self.criterion(preds, yolo_targets)
            loss.backward()
            self.optimizer.step()

            # Evaluate one batch for reporting
            if iteration % 10 == 0:
                with torch.no_grad():
                    predictions = self.pred_converter.convert_preds_to_list(preds)
                    formatted = formatter.format(predictions, targets)
                    schema = SignedSchema(
                        issuer_id=issuer_id,
                        schema=formatted,
                        timestamp=time.time()
                    )
                    metrics_broker.notify(schema)

            # Save checkpoint
            if iteration % 100 == 0 and iteration > 0:
                torch.save(self.model.state_dict(), f"checkpoints/yolov11_{iteration}.pt")

        torch.save(self.model.state_dict(), "checkpoints/yolov11_final.pt")
        return "YOLOv11 training complete."