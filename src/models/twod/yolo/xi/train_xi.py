from typing import cast

from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.factories.network_dataset_stream_factory import NetworkDatasetStreamFactory
from src.data.dataset.streams.prefetcher import Prefetcher
from src.models.twod.yolo.xi.yoloxi_dataset import UltralyticsDataset
from src.models.twod.yolo.xi.yoloxi_eval_dataset import EvalUltralyticsDataset
from src.models.twod.yolo.xi.yoloxi_training_setup import TrainingSetup


def main():
    server_ip = "10.0.0.1"

    print("📡 Creating streaming pipelines...")
    train_stream = NetworkDatasetStreamFactory(server_ip, DatasetSplit.TRAIN).create_stream()
    val_stream = NetworkDatasetStreamFactory(server_ip, DatasetSplit.VAL).create_stream()

    print("🧠 Building Ultralytics-compatible datasets...")
    train_dataset = UltralyticsDataset(cast(Prefetcher, train_stream), batch_size=8, num_batches=6495)
    val_dataset = EvalUltralyticsDataset(cast(Prefetcher, val_stream), batch_size=8, num_batches=430)

    trainer = TrainingSetup(train_dataset, val_dataset)
    trainer.train()


if __name__ == "__main__":
    main()
