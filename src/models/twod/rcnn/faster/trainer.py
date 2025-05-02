from typing import Dict, Tuple

import torch
import os

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from src.utils.logging import console


class Trainer:
    """Trainer for faster-RCNN."""

    def __init__(self, dataloader: DataLoader, n_classes: int, lr: float = 0.005, momentum: float = 0.9,
                 weight_decay: float = 5e-4, output_dir: str = "faster_rcnn_outputs"):
        """
        Initializes a Trainer instance.

        Args:
            dataloader (torch.utils.data.DataLoader): data loader for loading training data
            n_classes (int): number of classes
            lr (float): learning rate to use for training
            momentum (float): momentum factor for optimizer
            weight_decay (float): weight decay factor for optimizer
            output_dir (str): directory for writing outputs
        """
        self._dataloader = dataloader
        self._n_classes = n_classes
        self._lr = lr
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._output_dir = output_dir

        self._model = self._create_model()

        self._writer = SummaryWriter(log_dir=f"{self._output_dir}/tensorboard")

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
        if ckpt_path is not None:
            start_epoch = self._load_ckpt(self._model, optimizer, device, ckpt_path)

        console.log("[bold cyan]Starting training...[/bold cyan]")

        for epoch in range(start_epoch, n_epochs):
            self._model.train()
            n_batches = 0
            total_loss = 0.0
            total_cls = 0.0
            total_box = 0.0
            total_obj = 0.0
            total_rpn = 0.0

            with tqdm(self._dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}") as pbar:
                for images, targets in pbar:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    loss_dict = self._model(images, targets)
                    loss = sum(loss for loss in loss_dict.values())

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                    optimizer.step()

                    cls_loss, box_loss, obj_loss, rpn_loss = self._get_losses(loss_dict)

                    total_loss += loss.item()
                    total_cls += cls_loss
                    total_box += box_loss
                    total_obj += obj_loss
                    total_rpn += rpn_loss
                    n_batches += 1

            avg_loss = total_loss / n_batches
            avg_cls = total_cls / n_batches
            avg_box = total_box / n_batches
            avg_obj = total_obj / n_batches
            avg_rpn = total_rpn / n_batches

            self._log_losses(avg_loss, avg_cls, avg_box, avg_obj, avg_rpn, epoch + 1)
            self._save_ckpt(epoch, self._model, optimizer)

    def _create_model(self) -> Module:
        """Creates a faster-RCNN model."""
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self._n_classes)

        return model

    @staticmethod
    def _load_ckpt(model: Module, optimizer: Optimizer, device: torch.device, ckpt_path: str) -> int:
        """Loads checkpoint from path."""
        start_epoch = 0

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"]
            console.log(f"[bold yellow]Resuming from checkpoint {ckpt_path} at epoch {start_epoch}[/bold yellow]")

        return start_epoch

    def _save_ckpt(self, epoch: int, model: Module, optimizer: Optimizer) -> None:
        """Saves a checkpoint for the current state of the model."""
        epoch_ckpt_path = os.path.join(self._output_dir, f"epoch{epoch + 1}.pth")
        last_ckpt_path = os.path.join(self._output_dir, f"last_ckpt.pth")

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, epoch_ckpt_path)

        torch.save({
            "epoch": epoch + 1,
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

    def _log_losses(self, total: float, cls: float, box: float, obj: float, rpn: float, epoch: int) -> None:
        """Logs the losses from training."""
        self._writer.add_scalar("train/loss_total", total, epoch)
        self._writer.add_scalar("train/loss_cls", cls, epoch)
        self._writer.add_scalar("train/loss_box_reg", box, epoch)
        self._writer.add_scalar("train/loss_obj", obj, epoch)
        self._writer.add_scalar("train/loss_rpn_box_reg", rpn, epoch)

        console.log(
            f"[bold green]Epoch {epoch + 1}[/bold green] "
            f"| [cyan]Total[/cyan]: {total:.4f} "
            f"| [magenta]Class[/magenta]: {cls:.4f} "
            f"| [yellow]Box[/yellow]: {box:.4f} "
            f"| [blue]Obj[/blue]: {obj:.4f} "
            f"| [red]RPN[/red]: {rpn:.4f}"
        )