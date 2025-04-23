# src/models/twod/yolo/viii/streaming_trainer_viii.py

import tempfile
import yaml
import os

from torch.utils.tensorboard import SummaryWriter
from ultralytics.models.yolo.detect import DetectionTrainer

from src.models.twod.yolo.viii.streaming_evaluator_viii import StreamingEvaluatorVIII


class YOLOv8StreamingTrainer(DetectionTrainer):
    def __init__(self, exp):
        self.exp = exp
        self.train_dl, self.val_dl = exp.get_dataloaders()
        self.dummy_data_yaml = self._create_dummy_data_yaml()
        log_dir = str(os.path.join(exp.save_dir, exp.name, "custom_logs"))
        self.writer = SummaryWriter(log_dir=log_dir)

        overrides = {
            "model": exp.model,
            "imgsz": exp.input_size[0],
            "epochs": exp.epochs,
            "device": exp.device,
            "project": exp.save_dir,
            "name": exp.name,
            "save_period": exp.eval_interval,
            "save": True,
            "data": self.dummy_data_yaml,
            "val": False,
            "exist_ok": True,
        }

        if exp.resume_ckpt:
            overrides["resume"] = True
            overrides["weights"] = exp.resume_ckpt

        super().__init__(overrides=overrides)

    def _create_dummy_data_yaml(self):
        tmp_dir = tempfile.mkdtemp(prefix="streaming_yolov8_")
        os.makedirs(os.path.join(tmp_dir, "fake"), exist_ok=True)
        yaml_data = {
            "train": os.path.join(tmp_dir, "fake"),
            "val": os.path.join(tmp_dir, "fake"),
            "nc": self.exp.num_classes,
            "names": ["tail-biting", "ear-biting", "belly-nosing", "tail-down"]
        }
        yaml_path = os.path.join(tmp_dir, "data.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f)
        return yaml_path

    def get_dataloader(self, dataset_path=None, batch_size=None, rank=0, mode="train"):
        return self.train_dl if mode == "train" else self.val_dl

    def plot_training_labels(self):
        print("⚠️ Skipping label plotting — streaming dataset has no static `.labels`")

    def plot_training_samples(self, batch, ni):
        print("⚠️ Skipping sample image plotting — no file paths in streaming mode")

    def train(self):
        print("🚀 Starting YOLOv8 streaming training...")
        super().train()  # runs the main training loop
        print("✅ Training complete. Running final evaluation...")
        self.validate()

    def validate(self):
        print("🔍 Running custom StreamingEvaluatorVIII...")
        evaluator = StreamingEvaluatorVIII(
            model=self.model.model,
            dataloader=self.val_dl,
            device=self.exp.device,
            num_classes=self.exp.num_classes
        )
        self.metrics = evaluator.evaluate()
        print(f"📊 Evaluation metrics: {self.metrics}")

        if self.writer:
            for k, v in self.metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"val/{k}", v, self.epoch)