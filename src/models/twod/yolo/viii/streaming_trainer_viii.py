import os
from ultralytics import YOLO
from ultralytics.utils.torch_utils import de_parallel

class YOLOv8StreamingTrainer:
    def __init__(self, exp, train_dl, val_dl):
        self.exp = exp
        self.train_dl = train_dl
        self.val_dl = val_dl

        self.model = YOLO(exp.model)
        if exp.resume_ckpt:
            print(f"🔁 Resuming from checkpoint: {exp.resume_ckpt}")
            self.model = YOLO(exp.resume_ckpt)

        self.model.model.args["imgsz"] = exp.input_size
        self.model.model.args["epochs"] = exp.epochs
        self.model.model.args["save_period"] = exp.eval_interval
        self.model.model.args["save"] = True

    def train(self):
        self.model.train(
            data=None,  # we use our own dataloaders
            dataloader=self.train_dl,
            val_dataloader=self.val_dl,
            epochs=self.exp.epochs,
            save_dir=self.exp.save_dir,
            name=self.exp.name,
            verbose=True,
        )
