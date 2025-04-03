from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import tempfile
import os
import yaml


class TrainingSetup:
    def __init__(self, dataset, model_path="yolo11m-obb.pt", log_dir="runs", epochs=300, imgsz=640):
        self.dataset = dataset
        self.model_path = model_path
        self.epochs = epochs
        self.imgsz = imgsz

        # Load YOLO model
        self.model = YOLO(self.model_path)

        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(log_dir, f"yolo11m_obb_{timestamp}")
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Create dummy YAML config
        self.data_yaml_path = self._create_dummy_yaml()

        # Hook for logging
        self.model.add_callback("on_fit_epoch_end", self._log_epoch_metrics)

    def _create_dummy_yaml(self):
        content = {
            "train": "unused/train",
            "val": "unused/val",
            "nc": 4,
            "names": ["tail-biting", "ear-biting", "belly-nosing", "tail-down"]
        }

        fd, path = tempfile.mkstemp(suffix=".yaml", prefix="ultra_data_")
        with os.fdopen(fd, 'w') as f:
            yaml.safe_dump(content, f)
        return path

    def _log_epoch_metrics(self, trainer):
        epoch = trainer.epoch
        metrics = trainer.metrics
        if not metrics:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"metrics/{key}", value, epoch)

    def train(self):
        # Create a temporary YAML file to bypass file-based dataset loading
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_yaml:
            yaml.dump({
                "train": "dummy",  # not used
                "val": "dummy",  # not used
                "names": ["tail-biting", "ear-biting", "belly-nosing", "tail-down"]
            }, temp_yaml)
            temp_yaml_path = temp_yaml.name

        overrides = {
            "model": self.model_path,
            "data": temp_yaml_path,  # now just a string path
            "train": self.dataset,  # actual dataset
            "val": self.dataset,  # actual dataset
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "project": self.log_dir,
            "name": "train",
            "save": True,
            "verbose": True,
        }

        results = self.model.train(**overrides)

        metrics = results.results_dict if hasattr(results, "results_dict") else {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"final_metrics/{key}", value, self.epochs)

        self.writer.flush()
        self.writer.close()
