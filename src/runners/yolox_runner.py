from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataclasses.compressed_annotated_frame import CompressedAnnotatedFrame
from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.factories.network_dataset_stream_factory import NetworkDatasetStreamFactory
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

def main():
    train_factory = NetworkDatasetStreamFactory(server_ip=SERVER_IP, split=DatasetSplit.TRAIN)
    val_factory = NetworkDatasetStreamFactory(server_ip=SERVER_IP, split=DatasetSplit.VAL)

    train_set = YOLOXDataset(stream_factory=train_factory, batch_size=8, n_batches=6125)
    val_set = YOLOXDataset(stream_factory=val_factory, batch_size=8, n_batches=10)

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