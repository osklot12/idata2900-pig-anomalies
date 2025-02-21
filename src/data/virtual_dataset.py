from src.data.dataset_source import DatasetSource


class VirtualDataset:
    """
    Organizes frames into train, validation, and test sets while
    mimicking a traditional dataset stored on disk.
    """

    def __init__(self, source: DatasetSource,
                 train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        if train_ratio + val_ratio + test_ratio != 1:
            raise ValueError("Train, validation, and test ratios must sum to 1.")

        self.source = source

        self.train_data = []
        self.val_data = []
        self.test_data = []

    def feed_frame(self, video_name: str, frame_index: int, frame: bytearray, is_last_frame: bool):
        """Receives a frame and assigns it to the appropriate dataset split."""
        

    def _shuffle_and_split(self):
        """Shuffles and splits the dataset into train, validation, and test sets."""

