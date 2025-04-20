from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.factories.network_dataset_stream_factory import NetworkDatasetStreamFactory
from src.data.processing.zlib_decompressor import ZlibDecompressor
from src.models.converters.yolox_batch_converter import YOLOXBatchConverter
from src.models.twod.yolo.x.yolox_dataset import YOLOXDataset
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass

SERVER_IP = "10.0.0.1"


def run_train_stream():
    factory = NetworkDatasetStreamFactory(
        server_ip=SERVER_IP,
        split=DatasetSplit.TRAIN
    )

    stream = factory.create_stream()

    converter = YOLOXBatchConverter()

    batch_size = 8

    total = 0
    tail_biting = 0
    ear_biting = 0
    belly_nosing = 0
    tail_down = 0

    # act
    try:
        while True:
            batch = []
            while len(batch) < batch_size:
                batch.append(stream.read())
            images, targets, _, _ = converter.convert(batch)
            for instance in batch:
                print(f"[Test] Got frame {instance.index} from {instance.source.source_id}")
                if instance.cls == NorsvinBehaviorClass.TAIL_BITING:
                    tail_biting += 1
                if instance.cls == NorsvinBehaviorClass.TAIL_DOWN:
                    tail_down += 1
                if instance.cls == NorsvinBehaviorClass.BELLY_NOSING:
                    belly_nosing += 1
                if instance.cls == NorsvinBehaviorClass.EAR_BITING:
                    ear_biting += 1
            total += 1
            print(f"[Test] Total instances: {total}")
            print(f"[Test] Bellynosing: {belly_nosing} ({belly_nosing / total * 100:.2f}%)")
            print(f"[Test] Tailbiting: {tail_biting} ({tail_biting / total * 100:.2f}%)")
            print(f"[Test] Earbiting: {ear_biting} ({ear_biting / total * 100:.2f}%)")
            print(f"[Test] Taildown: {tail_down} ({tail_down / total * 100:.2f}%)")
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
