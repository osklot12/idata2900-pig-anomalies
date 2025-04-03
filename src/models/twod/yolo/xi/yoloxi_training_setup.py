import tempfile
import yaml
import os
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ultralytics.models.yolo.obb import OBBTrainer


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
        # Step 1: Create dummy YAML + folders
        dummy_root = tempfile.mkdtemp(prefix="yolo_dummy_")
        os.makedirs(os.path.join(dummy_root, "train", "images"), exist_ok=True)
        os.makedirs(os.path.join(dummy_root, "val", "images"), exist_ok=True)

        dummy_yaml = {
            "train": os.path.join(dummy_root, "train"),
            "val": os.path.join(dummy_root, "val"),
            "nc": 4,
            "names": ["tail-biting", "ear-biting", "belly-nosing", "tail-down"]
        }

        dummy_yaml_path = os.path.join(dummy_root, "data.yaml")
        with open(dummy_yaml_path, "w") as f:
            yaml.safe_dump(dummy_yaml, f)

        # Step 2: Setup overrides
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

        # Step 3: Create trainer
        trainer = OBBTrainer(overrides=overrides)

        # Step 4: Patch dataloader logic
        def patched_get_dataloader(_, dataset_path, batch_size, rank=0, mode="train"):
            return DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=False,  # shuffle must be False for IterableDataset
                num_workers=0,
                drop_last=False
            )
        trainer.get_dataloader = patched_get_dataloader.__get__(trainer)

        # ✅ Step 5: Patch label plotting (fix AttributeError)
        def skip_plot_labels(self):
            print("⚠️ Skipping plot_training_labels() — in-memory dataset has no .labels.")
        trainer.plot_training_labels = skip_plot_labels.__get__(trainer)

        # Step 6: Logging
        trainer.add_callback("on_fit_epoch_end", self._log_epoch_metrics)

        # Step 7: Train
        trainer.train()

        # Final metrics
        metrics = getattr(trainer, "metrics", {})
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"final_metrics/{key}", value, self.epochs)

        self.writer.flush()
        self.writer.close()
