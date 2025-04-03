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
        data_config = {
            "train": self.dataset,  # pass custom Dataset here
            "val": self.dataset,  # use same or separate for val
            "nc": 4,
            "names": ["tail-biting", "belly-nosing", "ear-biting", "tail-down"]
        }

        # 🔥 Do NOT store data_config in any instance variables — YOLO will try to YAML dump them!
        results = self.model.train(
            data=data_config,  # this is allowed at runtime
            epochs=self.epochs,
            imgsz=self.imgsz,
            project=self.log_dir,
            name="train",
            save=True,
            verbose=True
        )

        metrics = results.results_dict if hasattr(results, "results_dict") else {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"final_metrics/{key}", value, self.epochs)

        self.writer.flush()
        self.writer.close()
