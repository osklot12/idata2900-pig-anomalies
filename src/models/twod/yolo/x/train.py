import traceback

from src.data.dataset.dataset_split import DatasetSplit
from src.data.streaming.prefetchers.batch_prefetcher import BatchPrefetcher
from src.models.twod.yolo.x.streaming_trainer import StreamingTrainer
from src.network.client.simple_network_client import SimpleNetworkClient
from src.network.messages.serialization.pickle_message_deserializer import PickleMessageDeserializer
from src.network.messages.serialization.pickle_message_serializer import PickleMessageSerializer
from src.network.network_frame_instance_provider import NetworkFrameInstanceProvider
from src.models.twod.yolo.x.exp import Exp
from src.models.twod.yolo.x.yolox_dataset import YOLOXDataset
from yolox.core.trainer import Trainer
import argparse


def main():
    server_ip = "10.0.0.1"

    train_client = SimpleNetworkClient(PickleMessageSerializer(), PickleMessageDeserializer())
    train_client.connect(server_ip)

    val_client = SimpleNetworkClient(PickleMessageSerializer(), PickleMessageDeserializer())
    val_client.connect(server_ip)

    train_provider = NetworkFrameInstanceProvider(train_client)
    val_provider = NetworkFrameInstanceProvider(val_client)

    train_prefetcher = BatchPrefetcher(train_provider, DatasetSplit.TRAIN, 8)
    val_prefetcher = BatchPrefetcher(val_provider, DatasetSplit.VAL, 8)

    train_prefetcher.run()
    val_prefetcher.run()

    train_set = YOLOXDataset(train_prefetcher)
    val_set = YOLOXDataset(val_prefetcher)

    exp = Exp(train_set=train_set, val_set=val_set)

    args = argparse.Namespace(
        batch_size=8,
        devices=1,
        resume=False,
        start_epoch=0,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        experiment_name=exp.exp_name,
        ckpt=None,
        fp16=False,
        fuse=False,
        cache=False,
        occupy=False,
        logger="tensorboard"
    )

    print(f"Using exp of type {type(exp)}")
    trainer = StreamingTrainer(exp, args)
    print("TRAINING!")
    try:
        trainer.train()
    except Exception as e:
        print("\n❌ Exception caught during training:")
        traceback.print_exc()


if __name__ == '__main__':
    main()