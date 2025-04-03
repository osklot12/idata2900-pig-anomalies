import argparse
import traceback

import torch
from yolox.core.trainer import Trainer
from src.models.twod.yolo.x.minimal.dummy_streaming_dataset import DummyStreamingDataset
from src.models.twod.yolo.x.minimal.exp import Exp

class StreamingTrainer(Trainer):
    def get_train_loader(self, batch_size):
        dataset = DummyStreamingDataset(num_batches=1000)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.exp.data_num_workers,
            pin_memory=True
        )
        return loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        # Dummy eval loader, could be same as train
        return self.get_train_loader(batch_size)

    def evaluate_and_save_model(self):
        print("Dummy eval: skipping actual evaluation.")

if __name__ == "__main__":
    args = argparse.Namespace(
        batch_size=4,
        devices=1,
        local_rank=0,
        machine_rank=0,
        num_machines=1,
        resume=False,
        ckpt=None,
        experiment_name="dummy_exp",
        exp_file=None,
        name=None,
        fp16=False,
        cache=False,
        fuse=False,
        seed=42,
        logger="tensorboard",
    )

    exp = Exp()
    trainer = StreamingTrainer(exp, args)
    try:
        trainer.train()
    except Exception as e:
        print("\n❌ Exception caught during training:")
        traceback.print_exc()