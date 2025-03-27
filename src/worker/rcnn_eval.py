import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import box_iou

from src.worker.ddp_utils import get_ddp_init_method


class RCNNEvaluator:
    def __init__(self, rank, world_size, queue, model_path):
        self.rank = rank
        self.world_size = world_size
        self.queue = queue
        self.model_path = model_path

    def setup(self):
        init_method = get_ddp_init_method()
        dist.init_process_group("gloo", init_method=init_method, rank=self.rank, world_size=self.world_size)
        self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")

    def cleanup(self):
        dist.destroy_process_group()

    def load_model(self):
        model = fasterrcnn_resnet50_fpn(weights=None)
        model.transform = GeneralizedRCNNTransform(
            min_size=640,
            max_size=640,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225]
        )
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.to(self.device)
        return DDP(model, device_ids=[self.rank] if torch.cuda.is_available() else None)

    def evaluate_loop(self, model):
        model.eval()
        print(f"[Rank {self.rank}] 🚀 Evaluator ready, waiting for data...")

        frame_count = 0
        iou_scores = []

        while True:
            try:
                print("[Evaluator] Pulling from queue...")
                image_tensor, target = self.queue.get(timeout=600)
            except Exception:
                print(f"[Rank {self.rank}] ⏳ Timeout — no more data. Exiting.")
                break

            image_tensor = image_tensor.to(self.device).unsqueeze(0)
            target = {k: v.to(self.device) for k, v in target.items()}

            with torch.no_grad():
                output = model(image_tensor)[0]

            pred_boxes = output["boxes"]
            true_boxes = target["boxes"]

            if len(true_boxes) == 0 or len(pred_boxes) == 0:
                iou_scores.append(0.0)
            else:
                ious = box_iou(pred_boxes, true_boxes)
                best_ious = ious.max(dim=0)[0]
                iou_scores.append(best_ious.mean().item())

            frame_count += 1
            print(f"[Rank {self.rank}] ✅ Frame {frame_count} evaluated | Mean IoU: {iou_scores[-1]:.4f}")

        if iou_scores:
            mean_iou = sum(iou_scores) / len(iou_scores)
            print(f"\n[Rank {self.rank}] 📊 Final Evaluation — Avg IoU over {frame_count} frames: {mean_iou:.4f}")
        else:
            print(f"[Rank {self.rank}] 😐 No frames evaluated.")

def ddp_eval_worker(rank, world_size, queue, model_path):
    evaluator = RCNNEvaluator(rank, world_size, queue, model_path)
    evaluator.setup()
    model = evaluator.load_model()

    try:
        evaluator.evaluate_loop(model)
    finally:
        evaluator.cleanup()


def launch_evaluation(world_size, queue, model_path):
    mp.spawn(ddp_eval_worker, args=(world_size, queue, model_path), nprocs=world_size, join=True)
