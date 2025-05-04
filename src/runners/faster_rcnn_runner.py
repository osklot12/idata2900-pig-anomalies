import math

from torch.utils.data import DataLoader

from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.factories.network_dataset_stream_factory import NetworkDatasetStreamFactory
from src.data.dataset.streams.providers.closing_stream_provider import ClosingStreamProvider
from src.data.dataset.streams.providers.reusable_stream_provider import ReusableStreamProvider
from src.data.pipeline.pipeline import Pipeline
from src.data.pipeline.preprocessor import Preprocessor
from src.data.processing.bbox_denormalizer_processor import BBoxDenormalizerProcessor
from src.data.processing.zlib_decompressor import ZlibDecompressor
from src.models.streaming_evaluator import StreamingEvaluator
from src.models.twod.rcnn.faster.streaming_dataset import StreamingDataset
from src.models.twod.rcnn.faster.trainer import Trainer
from src.utils.norsvin_dataset_config import NORSVIN_TRAIN_SET_SIZE

SERVER_IP = "10.0.0.1"

BATCH_SIZE = 1

OUTPUT_DIR = "faster_rcnn_outputs"

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    pipeline = Pipeline(Preprocessor(ZlibDecompressor())).then(Preprocessor(BBoxDenormalizerProcessor()))
    train_factory = NetworkDatasetStreamFactory(server_ip=SERVER_IP, split=DatasetSplit.TRAIN, pipeline=pipeline)
    val_factory = NetworkDatasetStreamFactory(server_ip=SERVER_IP, split=DatasetSplit.VAL, pipeline=pipeline)

    train_provider = ReusableStreamProvider(train_factory.create_stream())
    val_provider = ClosingStreamProvider(val_factory)

    dataset = StreamingDataset(train_provider, n_batches=math.ceil(10)) # NORSVIN_TRAIN_SET_SIZE / BATCH_SIZE
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    evaluator = StreamingEvaluator(
        stream_provider=val_provider,
        classes=["tail_biting", "ear_biting", "belly_nosing", "tail_down"],
        output_dir=OUTPUT_DIR
    )

    trainer = Trainer(
        dataloader=dataloader,
        n_classes=5,
        lr=0.0025,
        evaluator=evaluator,
        output_dir=OUTPUT_DIR,
        log_interval=100,
        eval_interval=1,
        class_shift=-1,
    )
    trainer.train(ckpt_path="faster_rcnn_outputs/epoch21.pth")


if __name__ == "__main__":
    main()
