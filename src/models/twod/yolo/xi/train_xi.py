from src.data.streaming.prefetchers.fake_batch_prefetcher import FakeBatchPrefetcher
from src.models.twod.yolo.xi.yoloxi_dataset import UltralyticsDataset
from src.models.twod.yolo.xi.yoloxi_training_setup import TrainingSetup


def main():
    # 🧪 Fake prefetcher instead of real server
    prefetcher = FakeBatchPrefetcher(batch_size=4)
    prefetcher.run()

    # 🧠 Streamed dataset from fake frames
    dataset = UltralyticsDataset(prefetcher)

    # 🏋️ Train the YOLOv11m-OBB model on it
    setup = TrainingSetup(dataset=dataset)
    setup.train()


if __name__ == "__main__":
    main()
