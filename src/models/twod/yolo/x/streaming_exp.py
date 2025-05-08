from typing import TypeVar, List

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
                 classes: List[str], iou_thresh: float = 0.5, freeze_backbone: bool = False):
        """
        Initializes an Exp instance.

        Args:
            train_stream_provider (StreamProvider[T]): provider of training set streams
            val_stream_provider (StreamProvider[T]): provider of validation set streams
            classes (List[str]): list of class names
            iou_thresh (float): IoU threshold for filtering predictions
            freeze_backbone (bool): whether to freeze backbone layers
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

        self.classes = classes
        self.iou_thresh = iou_thresh

        self.evaluator = StreamingEvaluator(
            stream_provider=self._val_stream_provider,
            classes=self.classes,
            iou_thresh=self.iou_thresh,
            nms=False,
            output_dir=f"YOLOX_outputs"
        )

        self.freeze_backbone = freeze_backbone

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img: str = None):
        dataset = StreamingDataset(
            stream_provider=self._train_stream_provider,
            batch_size=28,
            n_batches=1 #267
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