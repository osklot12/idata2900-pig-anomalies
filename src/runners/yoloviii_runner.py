# src/runners/yoloviii_runner.py

import argparse
import os
import traceback

from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.factories.network_dataset_stream_factory import NetworkDatasetStreamFactory
from src.data.dataset.streams.providers.closing_stream_provider import ClosingStreamProvider
from src.data.dataset.streams.providers.reusable_stream_provider import ReusableStreamProvider
from src.models.twod.yolo.ultralytics.streaming_exp import YOLOXIStreamingExp
from src.models.twod.yolo.ultralytics.streaming_trainer_xi import YOLOXIStreamingTrainer

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    # parser.add_argument("--ckpt", type=str, default="runs/streaming_yolov8/weights/last.pt", help="Path to checkpoint")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device")
    args = parser.parse_args()

    print("📡 Setting up data streams...")
    train_factory = NetworkDatasetStreamFactory("10.0.0.1", DatasetSplit.TRAIN)
    val_factory = NetworkDatasetStreamFactory("10.0.0.1", DatasetSplit.VAL)

    train_provider = ReusableStreamProvider(train_factory.create_stream())
    val_provider = ClosingStreamProvider(val_factory)

    print("🧪 Preparing YOLOv8 experiment...")
    exp = YOLOXIStreamingExp(train_provider, val_provider, batch_size=args.batch, epochs=args.epochs, device=args.device)

   # if args.resume:
   #     exp.resume_ckpt = os.path.join(exp.save_dir, "weights", "last.pt")
   #     print(f"🔁 Resuming from checkpoint: {args.ckpt}")

    print("🚀 Starting trainer...")
    trainer = YOLOXIStreamingTrainer(exp)

    try:
        trainer.train()
    except Exception:
        print("\n💥 Training crashed:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
