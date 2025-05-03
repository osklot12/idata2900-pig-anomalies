import tempfile

import torch
import yaml
import os
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn import DetectionModel
from ultralytics.nn.modules import Detect

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

        print(f"[DEBUG] Type of self.model: {type(self.model)}")

        # 🧠 Step 1: Load actual model if it's still a string
        if isinstance(self.model, str):
            print("[DEBUG] Loading model from path...")
            self.model = YOLO(self.model).model
        elif hasattr(self.model, "model"):  # YOLO wrapper
            print("[DEBUG] Unwrapping YOLO wrapper...")
            self.model = self.model.model

        # 🧠 Step 2: Replace Detect head for 4 classes
        print("[Trainer] Replacing YOLOv11 detect head with 4-class head...")

        from ultralytics.nn.modules.conv import Conv

        for i, m in enumerate(self.model):
            if isinstance(m, Detect):
                print("[DEBUG] Found Detect head.")
                # Fetch input channels from first Conv layer in Detect head
                in_channels = m.cv2[0].in_channels if isinstance(m.cv2, torch.nn.Sequential) else m.cv2.in_channels
                new_head = Detect(nc=4, ch=[in_channels] * 3)
                new_head.names = ["tail-biting", "ear-biting", "belly-nosing", "tail-down"]
                new_head.initialize_biases()
                self.model[i] = new_head
                print("[Trainer] ✅ Head replaced with 4-class head.")
                break
        else:
            raise RuntimeError("❌ Detect head not found in YOLOv11 model.")

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
        print(f"[Trainer] Starting evaluation at epoch {self.epoch}")

        # If self.model is a path, load it
        if isinstance(self.exp.model, str):
            print(f"[Trainer] Loading model from checkpoint: {self.exp.model}")
            yolo_wrapper = YOLO(self.exp.model)
            self.model = yolo_wrapper.model  # used for training
        else:
            print("[Trainer] Using existing model object.")
            yolo_wrapper = YOLO(self.exp.model.model) if hasattr(self.exp.model, "model") else YOLO(self.exp.model)

        # ✅ Ensure eval mode
        yolo_wrapper.eval()

        # Use correct wrapped model for predictor
        predictor = YOLOXIPredictor(yolo_wrapper, device=self.exp.device)

        evaluator = StreamingEvaluator(
            stream_provider=self.exp.val_stream_provider,
            classes=["tail-biting", "ear-biting", "belly-nosing", "tail-down"],
            iou_thresh=0.5,
            output_dir="yoloxi_outputs"
        )

        evaluator.evaluate(predictor, epoch=self.epoch)

        print("[Trainer] Evaluation complete.")
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
