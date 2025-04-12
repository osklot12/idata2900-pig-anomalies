import traceback

from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.prefetcher import Prefetcher
from src.models.twod.yolo.x.streaming_trainer import StreamingTrainer
from src.network.client.simple_network_client import SimpleNetworkClient
from src.network.messages.serialization.pickle_message_deserializer import PickleMessageDeserializer
from src.network.messages.serialization.pickle_message_serializer import PickleMessageSerializer
from src.data.dataset.streams.network_stream import NetworkStream
from src.models.twod.yolo.x.exp import Exp
from src.models.twod.yolo.x.yolox_dataset import YOLOXDataset
import argparse


def main():
    server_ip = "10.0.0.1"

    train_client = SimpleNetworkClient(PickleMessageSerializer(), PickleMessageDeserializer())
    train_client.connect(server_ip)

    val_client = SimpleNetworkClient(PickleMessageSerializer(), PickleMessageDeserializer())
    val_client.connect(server_ip)

    train_stream = NetworkStream(client=train_client, split=DatasetSplit.TRAIN)
    val_stream = NetworkStream(client=val_client, split=DatasetSplit.VAL)

    batch_size = 8
    train_prefetcher = Prefetcher(train_stream, batch_size, 8, fetch_timeout=200)
    val_prefetcher = Prefetcher(val_stream, batch_size, 8, fetch_timeout=200)

    train_prefetcher.run()
    val_prefetcher.run()

    train_set = YOLOXDataset(train_prefetcher, 6495)
    val_set = YOLOXDataset(val_prefetcher, 430)

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
        ckpt=None, # "YOLOX_outputs/streaming_yolox/latest_ckpt.pth",
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