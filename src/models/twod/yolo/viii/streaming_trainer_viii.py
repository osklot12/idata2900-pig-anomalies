from ultralytics.models.yolo.detect import DetectionTrainer


class YOLOv8StreamingTrainer:
    def __init__(self, exp, train_dl, val_dl):
        self.exp = exp
        self.train_dl = train_dl
        self.val_dl = val_dl

    def train(self):
        class StreamingTrainer(DetectionTrainer):
            def get_dataloader(self, dataset_path=None, batch_size=None, rank=0, mode="train"):
                return self.train_dl if mode == "train" else self.val_dl

        overrides = {
            "model": self.exp.model,
            "imgsz": self.exp.input_size[0],
            "epochs": self.exp.epochs,
            "save_period": self.exp.eval_interval,
            "save": True,
            "project": self.exp.save_dir,
            "name": self.exp.name,
            "device": getattr(self.exp, "device", "cuda:0"),
            "data": {
                "names": ["tail-biting", "ear-biting", "belly-nosing", "tail-down"],
                "nc": self.exp.num_classes
            },
        }

        if self.exp.resume_ckpt:
            overrides["resume"] = self.exp.resume_ckpt

        print("🧠 Launching YOLOv8 training with custom dataloaders...")
        trainer = StreamingTrainer(overrides=overrides)
        trainer.train()
