from typing import TypeVar
from torch.utils.data import DataLoader
from src.data.dataset.streams.providers.stream_provider import StreamProvider
from src.models.twod.yolo.viii.yoloviii_dataset import StreamingDataset

T = TypeVar("T")

class YOLOv8StreamingExp:
    def __init__(self, train_stream_provider: StreamProvider, val_stream_provider: StreamProvider, batch_size=8, epochs=300, device="cuda:0"):
        print("🔧 YOLOv8StreamingExp setup:")
        print(f"  ├─ Batch size: {batch_size}")
        print(f"  ├─ Epochs: {epochs}")
        print(f"  └─ Device: {device}")

        self.train_stream_provider = train_stream_provider
        self.val_stream_provider = val_stream_provider
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device

        self.num_classes = 4
        self.input_size = (640, 640)
        self.model = "yolo11n.pt"
        self.name = "streaming_yolov8"
        self.save_dir = f"runs/{self.name}"
        self.eval_interval = 1
        self.resume_ckpt = self.model  # You can set this externally before training

    def get_train_loader(self):
        print("📥 Building dataloaders...")
        train_ds = StreamingDataset(
            self.train_stream_provider, batch_size=8, n_batches=10  #937
        )

        return DataLoader(train_ds, batch_size=None, num_workers=0, pin_memory=True)


    def get_val_loader(self):
        print("Building val loader...")

        val_ds = StreamingDataset(
            self.val_stream_provider, batch_size=8, n_batches=10 #430
        )

        return DataLoader(val_ds, batch_size=None, num_workers=0, pin_memory=True)
