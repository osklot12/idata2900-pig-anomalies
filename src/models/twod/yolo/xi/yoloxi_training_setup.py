import tempfile

import yaml
from ultralytics.models.yolo.obb import OBBTrainer
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


class TrainingSetup:
    def __init__(self, dataset, model_path="yolo11m-obb.pt", log_dir="runs", epochs=300, imgsz=640):
        self.dataset = dataset
        self.model_path = model_path
        self.epochs = epochs
        self.imgsz = imgsz

        # TensorBoard logging
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(log_dir, f"yolo11m_obb_{timestamp}")
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def _log_epoch_metrics(self, trainer):
        epoch = trainer.epoch
        metrics = getattr(trainer, "metrics", {})
        if not metrics:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"metrics/{key}", value, epoch)

    def train(self):
        # Create minimal dummy YAML file (YOLO just needs it to exist)
        dummy_yaml = {
            "train": "unused/train",
            "val": "unused/val",
            "nc": 4,
            "names": ["tail-biting", "ear-biting", "belly-nosing", "tail-down"]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(dummy_yaml, f)
            dummy_yaml_path = f.name

        overrides = {
            "model": self.model_path,
            "data": dummy_yaml_path,
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "project": self.log_dir,
            "name": "train",
            "save": True,
            "verbose": True,
        }

        # Now create trainer AFTER dummy file exists
        trainer = OBBTrainer(overrides=overrides)
        trainer.trainset = self.dataset
        trainer.testset = self.dataset  # optional
        trainer.add_callback("on_fit_epoch_end", self._log_epoch_metrics)

        # Train
        trainer.train()

        # Final metrics logging
        metrics = getattr(trainer, "metrics", {})
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"final_metrics/{key}", value, self.epochs)

        self.writer.flush()
        self.writer.close()