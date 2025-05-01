import tempfile
import yaml
import os
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import model_info

from src.models.twod.yolo.viii.batch_visualizer import visualize_batch_input
from src.models.twod.yolo.viii.streaming_evaluator_viii import StreamingEvaluatorVIII


class YOLOv8StreamingTrainer(DetectionTrainer):
    def __init__(self, exp):
        self.exp = exp
        self.train_dl, self.val_dl = exp.get_dataloaders()
        self.dummy_data_yaml = self._create_dummy_data_yaml()
        self._step_counter = 0

        log_dir = os.path.join(exp.save_dir, exp.name, "custom_logs")
        self.writer  = SummaryWriter(log_dir=log_dir)

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
            "exist_ok": True,
        }

        if exp.resume_ckpt:
            overrides["resume"] = False
            overrides["model"] = exp.resume_ckpt

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

    def get_validator(self):
        class CustomValidatorWrapper:
            def __init__(self, trainer):
                self.trainer = trainer
                self._metrics = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "mAP": 0.0,
                    "iou_loss": 0.0,
                    "conf_loss": 0.0,
                    "cls_loss": 0.0,
                    "total_loss": 0.0,
                    "fitness": 0.0,
                }

            @property
            def metrics(self):
                class MetricDict(dict):
                    @property
                    def keys(self_inner):  # Ultralytics expects `.metrics.keys` to be a list
                        return list(super(MetricDict, self_inner).keys())
                return MetricDict(self._metrics)

            @metrics.setter
            def metrics(self, value):
                self._metrics = value

            def __call__(self, *args, **kwargs):
                print("🔍 Running custom StreamingEvaluatorVIII from overridden validator...")
                evaluator = StreamingEvaluatorVIII(
                    model=self.trainer.model,
                    dataloader=self.trainer.val_dl,
                    device=self.trainer.exp.device,
                    num_classes=self.trainer.exp.num_classes,
                    writer=self.trainer.writer,
                    epoch=self.trainer.epoch
                )
                self.metrics = evaluator.evaluate()

                if self.trainer.writer:
                    for k, v in self.metrics.items():
                        if isinstance(v, (int, float)):
                            self.trainer.writer.add_scalar(f"val/{k}", v, self.trainer.epoch)

                return self.metrics

        return CustomValidatorWrapper(self)

    def validate(self):
        print("🔍 Running custom StreamingEvaluatorVIII...")

        if isinstance(self.model, str):
            print(f"📦 Loading model from checkpoint: {self.model}")
            self.model = YOLO(self.model)

            print(self.model.names)

        val_batch = next(iter(self.val_dl))
        visualize_batch_input(
            images=val_batch["img"],
            bboxes=val_batch["bboxes"],
            cls=val_batch["cls"],
            batch_idx=val_batch["batch_idx"],
            save_dir="./input_visuals/val",
            prefix=f"val_epoch{self.epoch}"
        )

        evaluator = StreamingEvaluatorVIII(
            model=self.model,
            dataloader=self.val_dl,
            device=self.exp.device,
            num_classes=self.exp.num_classes
        )
        self.metrics = evaluator.evaluate()

        metrics = self.metrics or {}
        fitness = (
                0.1 * metrics.get("recall", 0.0) +
                0.9 * metrics.get("mAP", 0.0)
        )

        if self.writer:
            for k, v in self.metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"val/{k}", v, self.epoch)

        return self.metrics, fitness

    def run_callbacks(self, event: str):
        super().run_callbacks(event)
        if event == "on_train_epoch_end":
            if self.writer and hasattr(self, "loss_items"):
                loss_names = ["box", "cls", "dfl"]
                for i, name in enumerate(loss_names):
                    self.writer.add_scalar(f"train/loss_{name}", self.loss_items[i], self.epoch)
                self.writer.add_scalar("train/loss_total", sum(self.loss_items), self.epoch)

    def train_batch(self, batch):
        # 🔍 Visualize input
        if not hasattr(self, "_step_counter"):
            self._step_counter = 0
        if self._step_counter % 10 == 0:
            visualize_batch_input(
                images=batch["img"],
                bboxes=batch["bboxes"],
                cls=batch["cls"],
                batch_idx=batch["batch_idx"],
                save_dir="./input_visuals/train",
                prefix=f"train_epoch{self.epoch}_step{self._step_counter}"
            )
        self._step_counter += 1

        # 🚀 Continue with training logic
        return super().train_batch(batch)


    def plot_training_labels(self):
            print("⚠️ Skipping plot_training_labels — streaming dataset has no `.labels` attribute.")

    def plot_training_samples(self, batch, ni):
        print("⚠️ Skipping plot_training_samples — streaming dataset has no `im_file` field.")
