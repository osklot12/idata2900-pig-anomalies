import torch
from torch.utils.data import Dataset, DataLoader
from yolox.exp import Exp as BaseExp
from yolox.core.trainer import Trainer
import argparse
import numpy as np


class DummyDataset(Dataset):
    def __getitem__(self, idx):
        img = np.random.rand(3, 640, 640).astype(np.float32)
        target = np.array([[0, 100, 100, 200, 200]], dtype=np.float32)  # [cls, x1, y1, x2, y2]
        img_info = np.array([640, 640], dtype=np.float32)
        img_id = idx
        return img, target, img_info, img_id

    def __len__(self):
        return 100


class DummyExp(BaseExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.50
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.exp_name = "dummy_yolox"

        self.max_epoch = 1
        self.print_interval = 1
        self.eval_interval = 1
        self.data_num_workers = 0
        self.basic_lr_per_img = 0.00015625

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=None):
        return DataLoader(
            dataset=DummyDataset(),
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True
        )

    def get_eval_loader(self, batch_size, is_distributed, **kwargs):
        return self.get_data_loader(batch_size, is_distributed)

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        return None


def main():
    exp = DummyExp()
    args = argparse.Namespace(
        batch_size=4,
        devices=1,
        resume=False,
        start_epoch=0,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        experiment_name=exp.exp_name,
        ckpt=None,
        fp16=False,
        fuse=False
    )

    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == "__main__":
    main()