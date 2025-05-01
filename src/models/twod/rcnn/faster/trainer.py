import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm

from src.utils.logging import console


class Trainer:
    """Trainer for faster-RCNN."""

    def __init__(self, dataloader: DataLoader, n_classes: int, optimizer: torch.optim.Optimizer):
        """
        Initializes a Trainer instance.

        Args:
            dataloader (torch.utils.data.DataLoader): data loader for loading training data
            n_classes (int): number of classes
            optimizer (torch.optim.Optimizer): the optimizer to use for training
        """
        self._dataloader = dataloader
        self._n_classes = n_classes
        self._optimizer = optimizer

    def train(self, n_epochs: int = 300) -> None:
        """
        Trains the model.

        Args:
            n_epochs (int): number of epochs
        """
        if n_epochs < 1:
            raise ValueError("n_epochs must be greater than 0")

        model = fasterrcnn_resnet50_fpn(num_classes=self._n_classes)
        device = self._get_device()
        model.to(device)

        console.log("[bold cyan]Starting training...[/bold cyan]")

        for epoch in range(n_epochs):
            model.train()
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

                    loss_dict = model(images, targets)
                    loss = sum(loss for loss in loss_dict.values())

                    loss_classifier = loss_dict.get("loss_classifier", torch.tensor(0.0)).item()
                    loss_box_reg = loss_dict.get("loss_box_reg", torch.tensor(0.0)).item()
                    loss_objectness = loss_dict.get("loss_objectness", torch.tensor(0.0)).item()
                    loss_rpn_box_reg = loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).item()

                    pbar.set_postfix({
                        "cls": loss_classifier,
                        "box": loss_box_reg,
                        "obj": loss_objectness,
                        "rpn": loss_rpn_box_reg
                    })

                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()

                    total_loss += loss.item()
                    total_cls += loss_classifier
                    total_box += loss_box_reg
                    total_obj += loss_objectness
                    total_rpn += loss_rpn_box_reg
                    n_batches += 1

            avg_loss = total_loss / n_batches
            avg_cls = total_cls / n_batches
            avg_box = total_box / n_batches
            avg_obj = total_obj / n_batches
            avg_rpn = total_rpn / n_batches

            console.log(
                f"[bold green]Epoch {epoch + 1}/{n_epochs}[/bold green] "
                f"| [cyan]Total[/cyan]: {avg_loss:.4f} "
                f"| [magenta]Class[/magenta]: {avg_cls:.4f} "
                f"| [yellow]Box[/yellow]: {avg_box:.4f} "
                f"| [blue]Obj[/blue]: {avg_obj:.4f} "
                f"| [red]RPN[/red]: {avg_rpn:.4f}"
            )

    @staticmethod
    def _get_device() -> torch.device:
        """Returns the device to use for training."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")