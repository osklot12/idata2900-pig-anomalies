import traceback

from src.data.dataset.streams.factories.network_dataset_stream_factory import NetworkDatasetStreamFactory
from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.providers.closing_stream_provider import ClosingStreamProvider
from src.data.dataset.streams.providers.reusable_stream_provider import ReusableStreamProvider
from src.models.twod.yolo.x.streaming_trainer import StreamingTrainer
from src.models.twod.yolo.x.streaming_exp import StreamingExp
import argparse

SERVER_IP = "10.0.0.1"


def main():
    train_factory = NetworkDatasetStreamFactory(server_ip=SERVER_IP, split=DatasetSplit.TRAIN)
    val_factory = NetworkDatasetStreamFactory(server_ip=SERVER_IP, split=DatasetSplit.VAL)

    train_provider = ReusableStreamProvider(stream=train_factory.create_stream())
    val_provider = ClosingStreamProvider(stream_factory=val_factory)

    exp = StreamingExp(
        train_stream_provider=train_provider,
        val_stream_provider=val_provider,
        classes=["tail_biting", "ear_biting", "belly_nosing", "tail_down"],
        freeze_backbone=True,
        iou_thresh=0.1
    )

    args = argparse.Namespace(
        batch_size=28,
        devices=1,
        resume=False,
        start_epoch=0,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        experiment_name=exp.exp_name,
        ckpt="YOLOX_weights/yolo_s.pth",
        fp16=True,
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
