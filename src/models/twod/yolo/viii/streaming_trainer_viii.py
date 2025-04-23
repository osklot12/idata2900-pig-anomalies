# src/models/twod/yolo/viii/streaming_trainer_viii.py

from ultralytics.models.yolo.detect import DetectionTrainer
import tempfile, yaml, os

class YOLOv8StreamingTrainer(DetectionTrainer):
    def __init__(self, exp):
        self.exp = exp
        self.train_dl, self.val_dl = exp.get_dataloaders()
        self.dummy_data_yaml = self._create_dummy_data_yaml()
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
        }

        if exp.resume_ckpt:
            overrides["resume"] = exp.resume_ckpt

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
