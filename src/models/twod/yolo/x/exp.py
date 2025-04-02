from src.models.twod.yolo.x.yolox_dataset import YOLOXDataset
from yolox.exp import Exp as BaseExp
from torch.utils.data import DataLoader

class Exp(BaseExp):
    """Experimental configurations for YOLOX."""

    def __init__(self, train_set: YOLOXDataset, val_set: YOLOXDataset):
        """
        Initializes an Exp instance.

        Args:
            train_set (YOLOXDataset): the dataset to use for training
            val_set (YOLOXDataset): the dataset to use for evaluation
        """
        super().__init__()
        self._train_set = train_set
        self._val_set = val_set

        self.num_classes = 4
        self.depth = 0.33
        self.width = 0.50
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.max_epoch = 300
        self.eval_interval = 10
        self.data_num_workers = 0
        self.tensorboard_writer = True

        self.exp_name = "streaming_yolox"

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img: str = None):
        return DataLoader(
            dataset=self._train_set,
            batch_size=None,
            num_workers=0,
            pin_memory=True
        )

    def get_eval_loader(self, batch_size, is_distributed, **kwargs):
        return DataLoader(
            dataset=self._val_set,
            batch_size=None,
            num_workers=0,
            pin_memory=True
        )