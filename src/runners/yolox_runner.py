import traceback

from src.data.dataset.streams.factories.network_dataset_stream_factory import NetworkDatasetStreamFactory
from src.data.dataset.dataset_split import DatasetSplit
from src.models.twod.yolo.x.streaming_trainer import StreamingTrainer
from src.models.twod.yolo.x.streaming_exp import StreamingExp
import argparse

SERVER_IP = "10.0.0.1"


def main():
    train_factory = NetworkDatasetStreamFactory(server_ip=SERVER_IP, split=DatasetSplit.TRAIN)
    val_factory = NetworkDatasetStreamFactory(server_ip=SERVER_IP, split=DatasetSplit.VAL)

    exp = StreamingExp(train_stream_factory=train_factory, val_stream_factory=val_factory)

    args = argparse.Namespace(
        batch_size=8,
        devices=1,
        resume=True,
        start_epoch=None,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        experiment_name=exp.exp_name,
        ckpt="YOLOX_outputs/streaming_yolox/epoch_31_ckpt.pth",
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
        print("\n Exception caught during training:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
