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

    def validate(self):
        """Runs validation using the StreamingEvaluator."""
        print(f"[Trainer] Starting evaluation at epoch {self.epoch}")

        # If model is still a string path, load it
        if isinstance(self.model, str):
            print(f"[Trainer] Loading model from checkpoint: {self.model}")
            self.model = YOLO(self.model)
        else:
            print(f"[Trainer] Model is already loaded.")

        # Confirm model device
        print(f"[Trainer] Model device: {self.exp.device}")

        # Setup evaluator
        evaluator = StreamingEvaluator(
            stream_provider=self.exp.val_stream_provider,
            classes=["tail-biting", "ear-biting", "belly-nosing", "tail-down"],
            iou_thresh=0.5,
            output_dir="yoloxi_outputs"
        )

        # Wrap model in predictor
        predictor = YOLOXIPredictor(self.model, device=self.exp.device)
        print("[Trainer] Created predictor. Running evaluator...")

        # Run evaluation
        evaluator.evaluate(predictor, epoch=self.epoch)

        print("[Trainer] Evaluation complete.")

        # No metrics returned from evaluate(), so just return dummy
        return {}, 0.0

    def run_callbacks(self, event: str):
        """Logs losses to TensorBoard at the end of each training epoch."""
        super().run_callbacks(event)
        if event == "on_train_epoch_end" and self.writer and hasattr(self, "loss_items"):
            loss_names = ["box", "cls", "dfl"]
            for i, name in enumerate(loss_names):
                self.writer.add_scalar(f"train/loss_{name}", self.loss_items[i], self.epoch)
            self.writer.add_scalar("train/loss_total", sum(self.loss_items), self.epoch)

    def plot_training_labels(self):
        print("Skipping plot_training_labels — streaming dataset has no `.labels` attribute.")

    def plot_training_samples(self, batch, ni):
        print("Skipping plot_training_samples — streaming dataset has no `im_file` field.")
