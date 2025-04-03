import argparse
import traceback

import torch
from yolox.core.trainer import Trainer
from src.models.twod.yolo.x.minimal.dummy_streaming_dataset import DummyStreamingDataset
from src.models.twod.yolo.x.minimal.exp import Exp

class StreamingTrainer(Trainer):
    class StreamingTrainer(Trainer):
        def get_train_loader(self, batch_size):
            dataset = DummyStreamingDataset(num_batches=1000)

            def collate_fn(batch):
                images = torch.stack([sample["img"] for sample in batch])
                img_info = torch.stack([sample["img_info"] for sample in batch])
                img_ids = torch.stack([sample["img_id"] for sample in batch])

                all_targets = []
                for i, sample in enumerate(batch):
                    bboxes = sample["gt_bboxes"]
                    classes = sample["gt_classes"].unsqueeze(1).float()
                    img_idx = torch.full((bboxes.size(0), 1), i, dtype=torch.float32)
                    target = torch.cat([bboxes, classes, img_idx], dim=1)  # shape: [N, 6]
                    all_targets.append(target)

                targets = torch.cat(all_targets, dim=0) if all_targets else torch.zeros((0, 6), dtype=torch.float32)

                return images, targets, img_info, img_ids

            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=self.exp.data_num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )

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