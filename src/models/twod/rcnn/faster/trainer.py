from typing import Dict, Tuple, Optional

import torch
import os

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.models.streaming_evaluator import StreamingEvaluator
from src.models.twod.rcnn.faster.faster_rcnn_predictor import FasterRCNNPredictor
from src.utils.logging import console

CONF_THRESH = 0.5


class Trainer:
    """Trainer for faster-RCNN."""

    def __init__(self, dataloader: DataLoader, n_classes: int, evaluator: Optional[StreamingEvaluator] = None,
                 lr: float = 0.005, momentum: float = 0.9, weight_decay: float = 5e-4,
                 output_dir: str = "faster_rcnn_outputs", log_interval: int = 10, eval_interval: int = 2,
                 class_shift: int = 1):
        """
        Initializes a Trainer instance.

        Args:
            dataloader (torch.utils.data.DataLoader): data loader for loading training data
            n_classes (int): number of classes
            evaluator (Optional[StreamingEvaluator]): evaluator for evaluating the model
            lr (float): learning rate to use for training
            momentum (float): momentum factor for optimizer
            weight_decay (float): weight decay factor for optimizer
            output_dir (str): directory for writing outputs
            log_interval (int): the interval for logging
            eval_interval (int): the interval for evaluating the model
            class_shift (int): the shift for the class ids, defaults to 0 (no shift)
        """
        self._dataloader = dataloader
        self._n_classes = n_classes
        self._evaluator = evaluator
        self._lr = lr
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._output_dir = output_dir
        self._log_interval = log_interval
        self._eval_interval = eval_interval
        self._class_shift = class_shift

        self._model = self._create_model()

        self._writer = SummaryWriter(log_dir=f"{self._output_dir}/tensorboard")

        self._total_epoch_steps = len(dataloader)

    def train(self, n_epochs: int = 300, ckpt_path: str = None) -> None:
        """
        Trains the model.

        Args:
            n_epochs (int): number of epochs
            ckpt_path (str): path to checkpoint file to train from
        """
        if n_epochs < 1:
            raise ValueError("n_epochs must be greater than 0")

        os.makedirs(self._output_dir, exist_ok=True)

        optimizer = torch.optim.SGD(
            self._model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
        )

        device = self._get_device()
        self._model.to(device)

        start_epoch = 0
        global_step = 0
        if ckpt_path is not None:
            start_epoch, global_step = self._load_ckpt(self._model, optimizer, device, ckpt_path)

        console.log("[bold]Starting training...[/bold]")


        for epoch in range(start_epoch, n_epochs):
            self._model.train()

            for images, targets in self._dataloader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                for target in targets:
                    target["labels"] += self._class_shift

                loss_dict = self._model(images, targets)
                loss = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()

                cls_loss, box_loss, obj_loss, rpn_loss = self._get_losses(loss_dict)

                global_step += 1
                if global_step % self._log_interval == 0:
                    self._log_losses(
                        loss.item(), cls_loss, box_loss, obj_loss, rpn_loss, epoch + 1, n_epochs, global_step
                                     )

            self._save_ckpt(epoch, self._model, optimizer, global_step)

            if (epoch + 1) % self._eval_interval == 0:
                self._evaluate(device, epoch)

    def _evaluate(self, device: torch.device, epoch: int) -> None:
        """Evaluates the model if an evaluator is given."""
        if self._evaluator:
            console.log("[bold]Evaluating...[/bold]")
            was_training = self._model.training
            self._model.eval()

            predictor = FasterRCNNPredictor(self._model, device=device, conf_thresh=CONF_THRESH)
            self._evaluator.evaluate(predictor, epoch=epoch)

            if was_training:
                self._model.train()

    def _create_model(self) -> Module:
        """Creates a faster-RCNN model."""
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self._n_classes)

        return model

    @staticmethod
    def _load_ckpt(model: Module, optimizer: Optimizer, device: torch.device, ckpt_path: str) -> Tuple[int, int]:
        start_epoch = 0
        global_step = 0

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0)
            global_step = ckpt.get("global_step", 0)
            console.log(f"[bold yellow]Resuming from checkpoint {ckpt_path} at epoch {start_epoch}, step {global_step}[/bold yellow]")

        return start_epoch, global_step

    def _save_ckpt(self, epoch: int, model: Module, optimizer: Optimizer, global_step: int) -> None:
        """Saves a checkpoint for the current state of the model."""
        epoch_ckpt_path = os.path.join(self._output_dir, f"epoch{epoch + 1}.pth")
        last_ckpt_path = os.path.join(self._output_dir, f"last_ckpt.pth")

        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, epoch_ckpt_path)

        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, last_ckpt_path)

    @staticmethod
    def _get_device() -> torch.device:
        """Returns the device to use for training."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _get_losses(loss_dict: Dict[str, float]) -> Tuple[float, float, float, float]:
        """Extracts individual losses from the loss dictionary."""
        loss_classifier = loss_dict.get("loss_classifier", torch.tensor(0.0)).item()
        loss_box_reg = loss_dict.get("loss_box_reg", torch.tensor(0.0)).item()
        loss_objectness = loss_dict.get("loss_objectness", torch.tensor(0.0)).item()
        loss_rpn_box_reg = loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).item()

        return loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg

    def _log_losses(self, total: float, cls: float, box: float, obj: float, rpn: float, epoch:int, total_epochs:int,
                    step: int) -> None:
        """Logs the losses from training."""
        self._writer.add_scalar("train/loss_total", total, step)
        self._writer.add_scalar("train/loss_cls", cls, step)
        self._writer.add_scalar("train/loss_box_reg", box, step)
        self._writer.add_scalar("train/loss_obj", obj, step)
        self._writer.add_scalar("train/loss_rpn_box_reg", rpn, step)

        console.log(
            f"[bold green]Epoch[/bold green] {epoch}/{total_epochs} "
            f"| [bold green]Step[/bold green] {step % self._total_epoch_steps}/{self._total_epoch_steps} "
            f"| [cyan]Total[/cyan]: {total:.4f} "
            f"| [magenta]Class[/magenta]: {cls:.4f} "
            f"| [yellow]Box[/yellow]: {box:.4f} "
            f"| [blue]Obj[/blue]: {obj:.4f} "
            f"| [red]RPN[/red]: {rpn:.4f}"
        )