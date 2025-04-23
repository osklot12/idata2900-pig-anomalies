from typing import TypeVar
from torch.utils.data import DataLoader
from ultralytics.data.build import build_dataloader
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from src.models.twod.yolo.viii.yoloviii_dataset import YOLOv8StreamingDataset

T = TypeVar("T")

class YOLOv8StreamingExp:
    def __init__(self, train_stream_factory, val_stream_factory):
        self.train_stream_factory = train_stream_factory
        self.val_stream_factory = val_stream_factory

        self.num_classes = 4
        self.input_size = (640, 640)
        self.epochs = 300
        self.model = "yolov8m.pt"
        self.name = "streaming_yolov8"
        self.save_dir = f"runs/{self.name}"
        self.eval_interval = 1
        self.resume_ckpt = None  # Set externally

    def get_dataloaders(self):
        train_ds = YOLOv8StreamingDataset(self.train_stream_factory, batch_size=28, max_batches=1750)
        val_ds = YOLOv8StreamingDataset(self.val_stream_factory, batch_size=8, max_batches=431, eval_mode=True)

        return (
            DataLoader(train_ds, batch_size=None, num_workers=0, pin_memory=True),
            DataLoader(val_ds, batch_size=None, num_workers=0, pin_memory=True),
        )
