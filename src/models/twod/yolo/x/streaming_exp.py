from typing import TypeVar

from src.data.dataset.streams.factories.stream_factory import ClosableStreamFactory
from src.models.twod.yolo.x.yolox_dataset import YOLOXDataset
from yolox.exp import Exp as BaseExp
from torch.utils.data import DataLoader

# stream data type
T = TypeVar("T")


class StreamingExp(BaseExp):
    """Experimental configurations for YOLOX."""

    def __init__(self, train_stream_factory: ClosableStreamFactory[T], val_stream_factory: ClosableStreamFactory[T]):
        """
        Initializes an Exp instance.

        Args:
            train_set (YOLOXDataset): the dataset to use for training
            val_set (YOLOXDataset): the dataset to use for evaluation
        """
        super().__init__()
        self._train_stream_factory = train_stream_factory
        self._val_stream_factory = val_stream_factory

        self.num_classes = 4
        self.depth = 0.33
        self.width = 0.50
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.max_epoch = 300
        self.eval_interval = 1
        self.data_num_workers = 0
        self.tensorboard_writer = True
        self.save_history_ckpt = True

        self.mosaic_prob = 0.0
        self.mixup_prob = 0.0
        self.hsv_prob = 0.0
        self.flip_prob = 0.0
        self.enable_mixup = False

        self.exp_name = "streaming_yolox"

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img: str = None):
        dataset = YOLOXDataset(
            stream_factory=self._train_stream_factory,
            batch_size=28,
            n_batches=1750
        )
        return DataLoader(
            dataset=dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=True,
        )

    def get_eval_loader(self, batch_size, is_distributed, **kwargs):
        dataset = YOLOXDataset(
            stream_factory=self._val_stream_factory,
            batch_size=8,
            n_batches=431
        )
        return DataLoader(
            dataset=dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=True,
        )

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        return self.input_size