import math
import traceback

from src.data.dataset.streams.factories.network_dataset_stream_factory import NetworkDatasetStreamFactory
from src.data.dataset.dataset_split import DatasetSplit
from src.models.twod.yolo.x.streaming_trainer import StreamingTrainer
from src.models.twod.yolo.x.yolo_exp import YoloExp
from src.models.twod.yolo.x.yolox_dataset import YOLOXDataset
import argparse


SERVER_IP = "10.0.0.1"

LEN_VAL_SET = 3442

def main():
    train_factory = NetworkDatasetStreamFactory(server_ip=SERVER_IP, split=DatasetSplit.TRAIN)
    val_factory = NetworkDatasetStreamFactory(server_ip=SERVER_IP, split=DatasetSplit.VAL)

    exp = YoloExp(train_stream_factory=train_factory, val_stream_factory=val_factory)

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