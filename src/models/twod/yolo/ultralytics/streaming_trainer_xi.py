import tempfile
import yaml
import os
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import model_info

from src.models.twod.yolo.ultralytics.batch_visualizer import visualize_batch_input
from src.models.twod.yolo.ultralytics.yoloxi_evaluator import StreamingEvaluatorXI


class YOLOXIStreamingTrainer(DetectionTrainer):
    """
    A custom trainer class for YOLOv11 (XI) using a streaming dataset.

    This class overrides the Ultralytics DetectionTrainer to integrate
    a streaming evaluation setup and TensorBoard logging.
    """

    def __init__(self, exp):
        """
        Initializes the streaming trainer with the given experiment configuration.

        Args:
            exp: Experiment configuration object containing dataloaders, model info, etc.
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
        """
        Creates a temporary dummy data.yaml file to satisfy the Ultralytics config format.

        Returns:
            str: Path to the created YAML file.
        """
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
        """
        Returns the dataloader depending on the training or validation mode.

        Args:
            mode (str): Either "train" or "val".

        Returns:
            DataLoader: The appropriate dataloader.
        """
        return self.train_dl if mode == "train" else self.val_dl

    def get_validator(self):
        """
        Returns a custom validator wrapper that integrates the streaming evaluation.

        Returns:
            Callable: Validator callable that computes metrics.
        """
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
                    def keys(self_inner):
                        return list(super(MetricDict, self_inner).keys())
                return MetricDict(self._metrics)

            @metrics.setter
            def metrics(self, value):
                self._metrics = value

            def __call__(self, *args, **kwargs):
                print("Running custom StreamingEvaluator...")
                evaluator = StreamingEvaluatorXI(
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
        """
        Runs validation using the custom streaming evaluator.

        Returns:
            Tuple[Dict[str, float], float]: Metrics dictionary and fitness score.
        """
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

        evaluator = StreamingEvaluatorXI(
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
        """
        Extends the default callback behavior to log training loss to TensorBoard.

        Args:
            event (str): Callback event name.
        """
        super().run_callbacks(event)
        if event == "on_train_epoch_end":
            if self.writer and hasattr(self, "loss_items"):
                loss_names = ["box", "cls", "dfl"]
                for i, name in enumerate(loss_names):
                    self.writer.add_scalar(f"train/loss_{name}", self.loss_items[i], self.epoch)
                self.writer.add_scalar("train/loss_total", sum(self.loss_items), self.epoch)

    def train_batch(self, batch):
        """
        Trains a single batch and optionally logs the input images.

        Args:
            batch (dict): A single training batch.

        Returns:
            dict: Loss values.
        """
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
        """
        Disabled: Streaming dataset does not contain label files.
        """
        print("Skipping plot_training_labels — streaming dataset has no `.labels` attribute.")

    def plot_training_samples(self, batch, ni):
        """
        Disabled: Streaming dataset does not contain image file metadata.

        Args:
            batch (dict): A training batch.
            ni (int): Image index.
        """
        print("Skipping plot_training_samples — streaming dataset has no `im_file` field.")
