# src/worker/rcnn_trainer.py

import torch
import os
import queue
import torchvision
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from src.worker.data_queue import data_queue


class RCNNTrainer:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

    def setup(self):
        dist.init_process_group("gloo", rank=self.rank, world_size=self.world_size)
        self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")

    def cleanup(self):
        dist.destroy_process_group()

    def create_model(self):
        model = fasterrcnn_resnet50_fpn(pretrained=True)

        # Replace default transform with one that fixes input size to 640x640
        model.transform = GeneralizedRCNNTransform(
            min_size=640,
            max_size=640,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225]
        )

        model.to(self.device)
        return DDP(model, device_ids=[self.rank] if torch.cuda.is_available() else None)

    def train_loop(self, model):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        model.train()

        save_dir = "checkpoints"
        os.makedirs(save_dir, exist_ok=True)

        iteration = 0

        while True:
            try:
                # Wait max 15 seconds for new data
                item = data_queue.get(timeout=15)
            except queue.Empty:
                print(f"[Rank {self.rank}] No data received for 15s — assuming done. Saving model and exiting.")
                torch.save(model.module.state_dict(), os.path.join(save_dir, "model_final.pt"))
                break

            images, targets = item
            images = [images.to(self.device)]
            targets = [{k: v.to(self.device) for k, v in targets.items()}]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            print(f"[Rank {self.rank}] Iter {iteration}, Loss: {losses.item():.4f}")

            if iteration % 100 == 0:
                ckpt_path = os.path.join(save_dir, f"model_checkpoint_{iteration}.pt")
                torch.save(model.module.state_dict(), ckpt_path)
                print(f"[Rank {self.rank}] Saved checkpoint at iter {iteration}")

            iteration += 1


def ddp_worker(rank, world_size):
    trainer = RCNNTrainer(rank, world_size)
    trainer.setup()
    model = trainer.create_model()

    try:
        trainer.train_loop(model)
    finally:
        trainer.cleanup()


def launch_training(world_size):
    mp.spawn(ddp_worker, args=(world_size,), nprocs=world_size, join=True)
