import os
import torch
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from src.models.converters.batch_converter import BatchConverter
from src.models.converters.target_converter import TargetConverter
from src.data.streaming.prefetchers.batch_prefetcher import BatchPrefetcher

class YOLOv11Trainer:
    def __init__(self, prefetcher: BatchPrefetcher, model_variant: str = "yolo11m-obb.pt", epochs: int = 50):
        self.prefetcher = prefetcher
        self.model_variant = model_variant
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo = YOLO(self.model_variant)  # Using the YOLO model
        self.model = None
        self.optimizer = None
        self.batch_converter = BatchConverter(self.device)  # Converts batches to tensors
        self.target_converter = TargetConverter(self.device)

    def setup(self):
        self.model = self.yolo.model
        self.model.to(self.device)
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def train(self) -> str:
        self.setup()
        os.makedirs("checkpoints", exist_ok=True)

        for iteration in range(self.epochs):
            batch = self.prefetcher.get()  # Get a batch
            images_list, targets = self.batch_converter.convert_to_tuple_of_tensors(batch)  # Convert batch to tensor format
            images_tensor = torch.stack(images_list).to(self.device)

            self.optimizer.zero_grad()

            # Convert targets using the TargetConverter
            yolo_targets = self.target_converter.convert_to_tensors(targets, images_list)

            # Compute the loss
            preds = self.model(images_tensor)  # Get predictions from the model
            loss = self.model.loss(preds, yolo_targets)  # Calculate loss

            loss.backward()
            self.optimizer.step()

            # Optionally, save model checkpoint every 100 iterations
            if iteration % 100 == 0 and iteration > 0:
                torch.save(self.model.state_dict(), f"checkpoints/yolov11_{iteration}.pt")

        torch.save(self.model.state_dict(), "checkpoints/yolov11_final.pt")
        return "YOLOv11 training complete."
