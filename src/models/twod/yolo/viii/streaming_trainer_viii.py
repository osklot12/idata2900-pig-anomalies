import tempfile
import os
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


class YOLOv8StreamingTrainer:
    def __init__(self, exp, train_dl, val_dl):
        self.exp = exp
        self.train_dl = train_dl
        self.val_dl = val_dl

        # Create a dummy data.yaml just to satisfy Ultralytics
        self.dummy_data_yaml = self._create_dummy_data_yaml()

        if exp.resume_ckpt:
            print(f"🔁 Resuming from checkpoint: {exp.resume_ckpt}")
            self.model = YOLO(exp.resume_ckpt)
        else:
            self.model = YOLO(exp.model)

    def _create_dummy_data_yaml(self):
        dummy_root = tempfile.mkdtemp(prefix="yolo_streaming_")
        os.makedirs(os.path.join(dummy_root, "fake", "images"), exist_ok=True)
        os.makedirs(os.path.join(dummy_root, "fake", "labels"), exist_ok=True)

        dummy_data = {
            "train": os.path.join(dummy_root, "fake"),
            "val": os.path.join(dummy_root, "fake"),
            "nc": self.exp.num_classes,
            "names": ["tail-biting", "ear-biting", "belly-nosing", "tail-down"]
        }

        yaml_path = os.path.join(dummy_root, "data.yaml")
        with open(yaml_path, "w") as f:
            yaml.safe_dump(dummy_data, f)

        return yaml_path

    def train(self):
        class StreamingTrainer(DetectionTrainer):
            def get_dataloader(self, dataset_path=None, batch_size=None, rank=0, mode="train"):
                return self.exp.train_dl if mode == "train" else self.exp.val_dl

        print("🧠 Launching YOLOv8 training with custom dataloaders...")
        overrides = {
            "model": self.exp.model,  # ✅ REQUIRED so it doesn't try to load None
            "imgsz": self.exp.input_size[0],
            "epochs": self.exp.epochs,
            "save_period": self.exp.eval_interval,
            "save": True,
            "project": self.exp.save_dir,
            "name": self.exp.name,
            "device": getattr(self.exp, "device", "cuda:0"),
            "data": self.dummy_data_yaml,
        }

        trainer = StreamingTrainer(overrides=overrides)
        trainer.train()

