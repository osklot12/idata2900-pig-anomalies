from yolox.exp import Exp as BaseExp
from torch.utils.data import DataLoader
from src.models.twod.yolo.x.minimal.dummy_streaming_dataset import DummyStreamingDataset

class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.50
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.data_num_workers = 0
        self.max_epoch = 1
        self.print_interval = 1
        self.eval_interval = 1
        self.exp_name = "dummy_exp"

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img: str = None):
        return DataLoader(
            DummyStreamingDataset(num_batches=100),
            batch_size=batch_size,
            num_workers=self.data_num_workers,
            pin_memory=True
        )