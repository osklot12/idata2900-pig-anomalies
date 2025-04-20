from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.factories.network_dataset_stream_factory import NetworkDatasetStreamFactory
from src.data.processing.zlib_decompressor import ZlibDecompressor
from src.models.converters.yolox_batch_converter import YOLOXBatchConverter
from src.models.twod.yolo.x.yolox_dataset import YOLOXDataset

SERVER_IP = "10.0.0.1"


def run_train_stream():
    factory = NetworkDatasetStreamFactory(
        server_ip=SERVER_IP,
        split=DatasetSplit.TRAIN
    )

    stream = factory.create_stream()

    decompressor = ZlibDecompressor()
    converter = YOLOXBatchConverter()

    batch_size = 8

    # act
    try:
        while True:
            batch = []
            while len(batch) < batch_size:
                batch.append(stream.read())
            images, targets, _, _ = converter.convert(batch)
            print(f"[Test] Actual targets: {[d.annotations for d in batch]}")
            print(f"[Test] Converted targets: {targets}")
            batch.clear()
    except KeyboardInterrupt:
        pass


def run_val_stream():
    factory = NetworkDatasetStreamFactory(server_ip=SERVER_IP, split=DatasetSplit.VAL)
    dataset = YOLOXDataset(stream_factory=factory, batch_size=8, n_batches=430)

    try:
        while True:
            for batch in dataset:
                print(f"[Test] Read batch of size {len(batch[0])}")

    except KeyboardInterrupt:
        print("[Test] Stopping...")


if __name__ == "__main__":
    run_train_stream()
