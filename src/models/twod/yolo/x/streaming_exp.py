from typing import TypeVar

from src.data.dataset.streams.providers.stream_provider import StreamProvider
from src.models.streaming_evaluator import StreamingEvaluator
from src.models.twod.yolo.x.streaming_dataset import StreamingDataset
from yolox.exp import Exp as BaseExp
from torch.utils.data import DataLoader

# stream data type
T = TypeVar("T")


class StreamingExp(BaseExp):
    """Experimental configurations for YOLOX."""

    def __init__(self, train_stream_provider: StreamProvider[T], val_stream_provider: StreamProvider[T],
                 evaluator: StreamingEvaluator, freeze_backbone: bool = False, iou_thresh: float = 0.5):
        """
        Initializes an Exp instance.

        Args:
            train_stream_provider (StreamProvider[T]): provider of training set streams
            val_stream_provider (StreamProvider[T]): provider of validation set streams
            evaluator (StreamingEvaluator): evaluator for evaluating model
            freeze_backbone (bool): whether to freeze backbone while training
            iou_thresh (float): iou threshold for filtering predictions
        """
        super().__init__()
        self._train_stream_provider = train_stream_provider
        self._val_stream_provider = val_stream_provider

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

        self.use_focal_loss = True
        self.focal_loss_gamma = 2.0
        self.focal_loss_alpha = 0.25

        self.freeze_backbone = freeze_backbone
        self.evaluator = evaluator
        self.iou_thresh = iou_thresh

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img: str = None):
        dataset = StreamingDataset(
            stream_provider=self._train_stream_provider,
            batch_size=28,
            n_batches=1 # 267
        )
        return DataLoader(
            dataset=dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=True,
        )

    def get_eval_loader(self, batch_size, is_distributed, **kwargs):
        dataset = StreamingDataset(
            stream_provider=self._val_stream_provider,
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