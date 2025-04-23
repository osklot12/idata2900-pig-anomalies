import os
import yaml
import tempfile
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


class YOLOv8StreamingTrainer:
    def __init__(self, exp, train_dl, val_dl):
        self.exp = exp
        self.train_dl = train_dl
        self.val_dl = val_dl

        if exp.resume_ckpt:
            print(f"🔁 Resuming from checkpoint: {exp.resume_ckpt}")
            self.model = YOLO(exp.resume_ckpt)
        else:
            self.model = YOLO(exp.model)

    def train(self):
        # ✅ Write dummy data.yaml
        dummy_data = {
            "names": ["tail-biting", "ear-biting", "belly-nosing", "tail-down"],
            "nc": self.exp.num_classes,
            "train": "fake/train",
            "val": "fake/val"
        }
        dummy_path = os.path.join(tempfile.mkdtemp(), "data.yaml")
        with open(dummy_path, "w") as f:
            yaml.safe_dump(dummy_data, f)

        # 🧠 Internal subclass that injects our custom dataloaders
        class StreamingTrainer(DetectionTrainer):
            def get_dataloader(self, dataset_path=None, batch_size=None, rank=0, mode="train"):
                return self.exp.train_dl if mode == "train" else self.exp.val_dl

        overrides = {
            "imgsz": self.exp.input_size[0],
            "epochs": self.exp.epochs,
            "save_period": self.exp.eval_interval,
            "save": True,
            "project": self.exp.save_dir,
            "name": self.exp.name,
            "device": getattr(self.exp, "device", "cuda:0"),
            "data": dummy_path  # ✅ path, not dict
        }

        if self.exp.resume_ckpt:
            overrides["resume"] = self.exp.resume_ckpt

        print("🧠 Launching YOLOv8 training with custom dataloaders...")
        trainer = StreamingTrainer(overrides=overrides)
        trainer.train()
