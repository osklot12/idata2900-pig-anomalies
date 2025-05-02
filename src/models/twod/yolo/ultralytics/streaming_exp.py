from typing import TypeVar
from torch.utils.data import DataLoader
from src.data.dataset.streams.providers.stream_provider import StreamProvider
from src.models.twod.yolo.ultralytics.ultralytics_dataset import StreamingDataset

T = TypeVar("T")


class YOLOXIStreamingExp:
    """
    Experiment setup for training YOLOv11 (XI) using a streaming dataset.

    This class defines the training and validation configuration including
    dataset providers, model path, input size, training device, and other metadata.
    """

    def __init__(self, train_stream_provider: StreamProvider, val_stream_provider: StreamProvider,
                 batch_size=8, epochs=300, device="cuda:0"):
        """
        Initializes a YOLOXIStreamingExp instance.

        Args:
            train_stream_provider (StreamProvider): Stream provider for training data.
            val_stream_provider (StreamProvider): Stream provider for validation data.
            batch_size (int): Number of samples per streamed batch.
            epochs (int): Number of training epochs.
            device (str): The device identifier (e.g., 'cuda:0' or 'cpu').
        """
        print("YOLOv11 StreamingExp setup:")
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
        self.name = "streaming_yolov11"
        self.save_dir = f"runs/{self.name}"
        self.eval_interval = 1
        self.resume_ckpt = self.model  # Can be externally overridden before training

    def get_train_loader(self) -> DataLoader:
        """
        Creates and returns a training DataLoader using the streaming dataset.

        Returns:
            DataLoader: Streaming DataLoader for training.
        """
        print("Building training DataLoader...")
        train_ds = StreamingDataset(
            self.train_stream_provider, batch_size=self.batch_size, n_batches=10  # Can be changed as needed
        )
        return DataLoader(train_ds, batch_size=None, num_workers=0, pin_memory=True)

    def get_val_loader(self) -> DataLoader:
        """
        Creates and returns a validation DataLoader using the streaming dataset.

        Returns:
            DataLoader: Streaming DataLoader for validation.
        """
        print("Building validation DataLoader...")
        val_ds = StreamingDataset(
            self.val_stream_provider, batch_size=self.batch_size, n_batches=10  # Can be changed as needed
        )
        return DataLoader(val_ds, batch_size=None, num_workers=0, pin_memory=True)
