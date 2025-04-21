from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.factories.network_dataset_stream_factory import NetworkDatasetStreamFactory
from src.data.dataset.streams.prefetcher import Prefetcher
from src.models.twod.yolo.xi.yoloxi_dataset import UltralyticsDataset
from src.models.twod.yolo.xi.yoloxi_eval_dataset import EvalUltralyticsDataset
from src.models.twod.yolo.xi.yoloxi_training_setup import TrainingSetup

def main():
    server_ip = "10.0.0.1"
    stream_buffer_size = 10  # You can adjust this buffer size

    # Create prefetching + pipelined data streams for training and validation
    train_stream = NetworkDatasetStreamFactory(server_ip, DatasetSplit.TRAIN).create_stream()
    val_stream = NetworkDatasetStreamFactory(server_ip, DatasetSplit.VAL).create_stream()

    print("🚀 Prefetchers + pipelines are now running.")

    print("🧠 Wrapping streams into model-compatible datasets...")
    train_dataset = UltralyticsDataset(train_stream, total_frames=6495)
    val_dataset = EvalUltralyticsDataset(val_stream, total_frames=6495)

    trainer = TrainingSetup(train_dataset, val_dataset)
    trainer.train()

if __name__ == "__main__":
    main()
