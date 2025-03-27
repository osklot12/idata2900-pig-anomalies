import multiprocessing

import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from src.worker.ddp_utils import get_ddp_init_method

class RCNNTrainer:
    def __init__(self, rank, world_size, queue: multiprocessing.Queue):
        self.rank = rank
        self.world_size = world_size
        self.queue = queue

    def setup(self):
        init_method = get_ddp_init_method()
        dist.init_process_group("gloo", init_method=init_method, rank=self.rank, world_size=self.world_size)
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
        try:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
            model.train()

            save_dir = "checkpoints"
            os.makedirs(save_dir, exist_ok=True)

            iteration = 0
            print(f"[Rank {self.rank}] 🚀 Trainer started, waiting for data...")

            while True:
                try:
                    print('Trying to access queue')
                    item = self.queue.get()
                    print(f"[Rank {self.rank}] 🧠 Pulled frame from queue")

                except self.queue.Empty:
                    print(f"[Rank {self.rank}] No data received for 10m — assuming done. Saving model and exiting.")
                    torch.save(model.module.state_dict(), os.path.join(save_dir, "model_final.pt"))
                    break

                try:
                    images, targets = item
                    images = [images.to(self.device)]
                    targets = [{k: v.to(self.device) for k, v in targets.items()}]

                    loss_dict = model(images, targets)
                    losses = torch.stack(list(loss_dict.values())).sum()

                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                    print(f"[Rank {self.rank}] Iter {iteration}, Loss: {losses.item():.4f}")

                    if iteration % 100 == 0:
                        ckpt_path = os.path.join(save_dir, f"model_checkpoint_{iteration}.pt")
                        torch.save(model.module.state_dict(), ckpt_path)
                        print(f"[Rank {self.rank}] Saved checkpoint at iter {iteration}")

                    iteration += 1
                except Exception as e:
                    print(f"[Rank {self.rank}] ❌ Error during training step: {e}")
        except Exception as e:
            print(f"[Rank {self.rank}] ❌ Trainer failed to start: {e}")


def ddp_worker(rank, world_size, queue):
    trainer = RCNNTrainer(rank, world_size, queue)
    trainer.setup()

    model = trainer.create_model()

    try:
        trainer.train_loop(model)
    finally:
        trainer.cleanup()


def launch_training(world_size, queue):
    mp.spawn(ddp_worker, args=(world_size, queue), nprocs=world_size, join=True)
