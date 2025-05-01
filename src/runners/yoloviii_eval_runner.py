# src/runners/yoloviii_eval_runner.py

import argparse
import traceback

from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.factories.network_dataset_stream_factory import NetworkDatasetStreamFactory
from src.models.twod.yolo.viii.streaming_exp_viii import YOLOv8StreamingExp
from src.models.twod.yolo.viii.streaming_trainer_viii import YOLOv8StreamingTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (e.g. best.pt)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run evaluation on")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for evaluation")
    args = parser.parse_args()

    print("📡 Setting up data streams...")
    val_factory = NetworkDatasetStreamFactory("10.0.0.1", DatasetSplit.VAL)

    print("🧪 Preparing YOLOv8 experiment (eval only)...")
    dummy_train_factory = NetworkDatasetStreamFactory("10.0.0.1", DatasetSplit.TRAIN)  # not used
    exp = YOLOv8StreamingExp(dummy_train_factory, val_factory, batch_size=args.batch, epochs=1, device=args.device)
    exp.resume_ckpt = args.ckpt

    print("🔍 Initializing trainer for evaluation...")
    trainer = YOLOv8StreamingTrainer(exp)
    trainer.epoch = 0  # for TensorBoard logging

    try:
        print("🧪 Starting evaluation...")
        trainer.validate()
    except Exception:
        print("\n💥 Evaluation crashed:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
