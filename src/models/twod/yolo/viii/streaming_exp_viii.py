from typing import TypeVar
from torch.utils.data import DataLoader
from src.models.twod.yolo.viii.yoloviii_dataset import YOLOv8StreamingDataset

T = TypeVar("T")

class YOLOv8StreamingExp:
    def __init__(self, train_stream_factory, val_stream_factory, batch_size=16, epochs=300, device="cuda:0"):
        print("🔧 YOLOv8StreamingExp setup:")
        print(f"  ├─ Batch size: {batch_size}")
        print(f"  ├─ Epochs: {epochs}")
        print(f"  └─ Device: {device}")

        self.train_stream_factory = train_stream_factory
        self.val_stream_factory = val_stream_factory
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device

        self.num_classes = 4
        self.input_size = (640, 640)
        self.model = "yolov8s.pt"
        self.name = "streaming_yolov8"
        self.save_dir = f"runs/{self.name}"
        self.eval_interval = 1
        self.resume_ckpt = None  # You can set this externally before training

    def get_dataloaders(self):
        print("📥 Building dataloaders...")
        train_ds = YOLOv8StreamingDataset(
            self.train_stream_factory, batch_size=self.batch_size, max_batches=467
        )
        val_ds = YOLOv8StreamingDataset(
            self.val_stream_factory, batch_size=self.batch_size, max_batches=215, eval_mode=True
        )

        train_dl = DataLoader(train_ds, batch_size=None, num_workers=0, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=None, num_workers=0, pin_memory=True)

        print("✅ Dataloaders created.")
        return train_dl, val_dl
