from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


class TrainingSetup:
    def __init__(self, dataset, model_path="yolo11m-obb.pt", log_dir="runs", epochs=300, imgsz=640):
        self.dataset = dataset
        self.model_path = model_path
        self.epochs = epochs
        self.imgsz = imgsz

        # Load Ultralytics YOLO model
        self.model = YOLO(self.model_path)

        # TensorBoard logging
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(log_dir, f"yolo11m_obb_{timestamp}")
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Hook to log per-epoch metrics
        self.model.add_callback("on_fit_epoch_end", self._log_epoch_metrics)

    def _log_epoch_metrics(self, trainer):
        epoch = trainer.epoch

        metrics = trainer.metrics
        if not metrics:
            return

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"metrics/{key}", value, epoch)

    def train(self):
        # This dictionary overrides internal YOLOTrainer args
        overrides = {
            "model": self.model_path,
            "data": None,  # Bypass file-based dataset
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "train": self.dataset,  # In-memory dataset
            "val": self.dataset,
            "project": self.log_dir,
            "name": "train",
            "save": True,
            "verbose": True,
        }

        # NOTE: Passing `trainer=YourCustomTrainer` won't work if it's not exposed internally
        results = self.model.train(**overrides)

        # Final metrics logging
        metrics = results.results_dict if hasattr(results, "results_dict") else {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"final_metrics/{key}", value, self.epochs)

        self.writer.flush()
        self.writer.close()
