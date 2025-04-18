from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataclasses.compressed_annotated_frame import CompressedAnnotatedFrame
from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.pipeline_stream import PipelineStream
from src.data.dataset.streams.prefetcher import Prefetcher
from src.data.dataset.streams.stream import Stream
from src.data.pipeline.pipeline import Pipeline
from src.data.pipeline.preprocessor import Preprocessor
from src.data.processing.zlib_decompressor import ZlibDecompressor
from src.models.twod.yolo.x.yolo_exp import YoloExp
from src.models.twod.yolo.x.yolox_dataset import YOLOXDataset
from src.network.client.simple_network_client import SimpleNetworkClient
from src.network.messages.serialization.pickle_message_deserializer import PickleMessageDeserializer
from src.network.messages.serialization.pickle_message_serializer import PickleMessageSerializer
from src.data.dataset.streams.network_stream import NetworkStream

SERVER_IP = "10.0.0.1"

def create_dataset_stream(split: DatasetSplit) -> Stream[AnnotatedFrame]:
    """Creates a dataset stream."""
    client = SimpleNetworkClient(PickleMessageSerializer(), PickleMessageDeserializer())
    client.connect(SERVER_IP)

    network_stream = NetworkStream(client=client, split=split, data_type=CompressedAnnotatedFrame)
    prefetcher = Prefetcher(network_stream)
    pipeline = Pipeline(Preprocessor(ZlibDecompressor()))
    stream = PipelineStream(source=prefetcher, pipeline=pipeline)

    prefetcher.run()

    return stream

def create_train_dataset() -> YOLOXDataset[AnnotatedFrame]:
    """Creates a dataset for training."""
    stream = create_dataset_stream(DatasetSplit.TRAIN)
    return YOLOXDataset(stream=stream, batch_size=8, n_batches=430)

def create_val_dataset() -> YOLOXDataset[AnnotatedFrame]:
    """Creates a dataset for validation."""
    stream = create_dataset_stream(DatasetSplit.VAL)
    return YOLOXDataset(stream=stream, batch_size=8, n_batches=6125)

def main():
    train_set = create_train_dataset()
    val_set = create_val_dataset()
    exp = YoloExp(train_set=train_set, val_set=val_set)

    args = argparse.Namespace(
        batch_size=8,
        devices=1,
        resume=False,
        start_epoch=0,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        experiment_name=exp.exp_name,
        ckpt="YOLOX_weights/yolox_s.pth",
        fp16=False,
        fuse=False,
        cache=False,
        occupy=False,
        logger="tensorboard"
    )

    trainer = StreamingTrainer(exp, args)
    try:
        trainer.train()
    except Exception as e:
        print("\n❌ Exception caught during training:")
        traceback.print_exc()

if __name__ == "__main__":
    main()