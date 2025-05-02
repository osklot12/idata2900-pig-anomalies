import tempfile
import yaml
import os
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from src.models.twod.yolo.ultralytics.batch_visualizer import visualize_batch_input
from src.models.twod.yolo.ultralytics.yoloxi_predictor import YOLOXIPredictor
from src.models.streaming_evaluator import StreamingEvaluator


class YOLOXIStreamingTrainer(DetectionTrainer):
    """
    A custom trainer for YOLOv11 (XI) using a streaming dataset.

    This class extends the Ultralytics DetectionTrainer and uses a custom StreamingEvaluator
    with a predictor interface. It logs losses to TensorBoard and saves training inputs.
    """

    def __init__(self, exp):
        """
        Initialize the trainer with experiment config.

        Args:
            exp: An experiment object providing model path, dataloaders, and training settings.
        """
        self.exp = exp
        self.train_dl = exp.get_train_loader()
        self.val_dl = exp.get_val_loader()
        self.dummy_data_yaml = self._create_dummy_data_yaml()
        self._step_counter = 0

        log_dir = os.path.join(exp.save_dir, exp.name, "custom_logs")
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
            "exist_ok": True,
        }

        if exp.resume_ckpt:
            overrides["resume"] = False
            overrides["model"] = exp.resume_ckpt

        super().__init__(overrides=overrides)

    def _create_dummy_data_yaml(self):
        """Creates a fake data.yaml file required by Ultralytics training loop."""
        tmp_dir = tempfile.mkdtemp(prefix="streaming_yolov11")
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
        """Return training or validation dataloader based on mode."""
        return self.train_dl if mode == "train" else self.val_dl

    def get_validator(self):
        """
        Returns a callable validator that wraps the StreamingEvaluator.

        This validator wraps the model in a YOLOXIPredictor before evaluation.
        """
        class CustomValidatorWrapper:
            def __init__(self, trainer):
                self.trainer = trainer
                self._metrics = {}

            @property
            def metrics(self):
                class MetricDict(dict):
                    @property
                    def keys(self_inner):
                        return list(super(MetricDict, self_inner).keys())
                return MetricDict(self._metrics)

            @metrics.setter
            def metrics(self, value):
                self._metrics = value

            def __call__(self, *args, **kwargs):
                predictor = YOLOXIPredictor(self.trainer.model, device=self.trainer.exp.device)
                evaluator = StreamingEvaluator(
                    model=predictor,
                    dataloader=self.trainer.val_dl,
                    device=self.trainer.exp.device,
                    num_classes=self.trainer.exp.num_classes,
                    writer=self.trainer.writer,
                    epoch=self.trainer.epoch
                )
                self.metrics = evaluator.evaluate()
                return self.metrics

        return CustomValidatorWrapper(self)

    def validate(self):
        """Runs validation using the StreamingEvaluator."""
        if isinstance(self.model, str):
            print(f"Loading model from checkpoint: {self.model}")
            self.model = YOLO(self.model)

        val_batch = next(iter(self.val_dl))
        visualize_batch_input(
            images=val_batch["img"],
            bboxes=val_batch["bboxes"],
            cls=val_batch["cls"],
            batch_idx=val_batch["batch_idx"],
            save_dir="./input_visuals/val",
            prefix=f"val_epoch{self.epoch}"
        )

        predictor = YOLOXIPredictor(self.model, device=self.exp.device)
        evaluator = StreamingEvaluator(
            model=predictor,
            dataloader=self.val_dl,
            device=self.exp.device,
            num_classes=self.exp.num_classes,
            writer=self.writer,
            epoch=self.epoch
        )
        self.metrics = evaluator.evaluate()
        fitness = (
            0.1 * self.metrics.get("recall", 0.0) +
            0.9 * self.metrics.get("mAP", 0.0)
        )

        return self.metrics, fitness

    def run_callbacks(self, event: str):
        """Logs losses to TensorBoard at the end of each training epoch."""
        super().run_callbacks(event)
        if event == "on_train_epoch_end" and self.writer and hasattr(self, "loss_items"):
            loss_names = ["box", "cls", "dfl"]
            for i, name in enumerate(loss_names):
                self.writer.add_scalar(f"train/loss_{name}", self.loss_items[i], self.epoch)
            self.writer.add_scalar("train/loss_total", sum(self.loss_items), self.epoch)

    def train_batch(self, batch):
        """Logs training inputs every 10 steps and trains the batch."""
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
        return super().train_batch(batch)

    def plot_training_labels(self):
        print("Skipping plot_training_labels — streaming dataset has no `.labels` attribute.")

    def plot_training_samples(self, batch, ni):
        print("Skipping plot_training_samples — streaming dataset has no `im_file` field.")
