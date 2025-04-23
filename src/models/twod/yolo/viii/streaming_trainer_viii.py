import os
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
        # Define a custom trainer class to use our streaming dataloaders
        class StreamingTrainer(DetectionTrainer):
            def get_dataloader(self, dataset_path=None, batch_size=None, rank=0, mode="train"):
                return self.args.train_dl if mode == "train" else self.args.val_dl

        overrides = {
            "imgsz": self.exp.input_size[0],
            "epochs": self.exp.epochs,
            "save_period": self.exp.eval_interval,
            "save": True,
            "project": self.exp.save_dir,
            "name": self.exp.name,
            "device": getattr(self.exp, "device", "cuda:0"),
            "train_dl": self.train_dl,
            "val_dl": self.val_dl,
            "data": {
                "names": ["tail-biting", "ear-biting", "belly-nosing", "tail-down"],
                "nc": self.exp.num_classes
            }
        }

        if self.exp.resume_ckpt:
            overrides["resume"] = self.exp.resume_ckpt

        print("🧠 Launching YOLOv8 training with streaming dataloaders...")
        self.model.train(
            overrides=overrides,
            trainer=StreamingTrainer
        )
