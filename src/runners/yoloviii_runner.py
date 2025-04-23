import argparse
import traceback

from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.factories.network_dataset_stream_factory import NetworkDatasetStreamFactory
from src.models.twod.yolo.viii.streaming_exp_viii import YOLOv8StreamingExp
from src.models.twod.yolo.viii.streaming_trainer_viii import YOLOv8StreamingTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--ckpt", type=str, default="runs/streaming_yolov8/weights/last.pt", help="Path to checkpoint")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device")
    args = parser.parse_args()

    print("📡 Setting up data streams...")
    train_factory = NetworkDatasetStreamFactory("10.0.0.1", DatasetSplit.TRAIN)
    val_factory = NetworkDatasetStreamFactory("10.0.0.1", DatasetSplit.VAL)

    print("🧪 Preparing YOLOv8 experiment...")
    exp = YOLOv8StreamingExp(train_factory, val_factory, batch_size=args.batch, epochs=args.epochs, device=args.device)

    if args.resume:
        exp.resume_ckpt = args.ckpt
        print(f"🔁 Resuming from checkpoint: {args.ckpt}")

    print("📦 Building dataloaders...")
    train_dl, val_dl = exp.get_dataloaders()

    print("🚀 Starting trainer...")
    trainer = YOLOv8StreamingTrainer(exp, train_dl, val_dl)

    try:
        trainer.train()
    except Exception:
        print("\n💥 Training crashed:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
