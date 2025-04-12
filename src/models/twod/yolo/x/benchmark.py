from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.prefetcher import Prefetcher
from src.models.twod.yolo.x.streaming_trainer import StreamingTrainer
from src.network.client.simple_network_client import SimpleNetworkClient
from src.network.messages.serialization.pickle_message_deserializer import PickleMessageDeserializer
from src.network.messages.serialization.pickle_message_serializer import PickleMessageSerializer
from src.network.network_frame_instance_provider import NetworkFrameInstanceProvider
from src.models.twod.yolo.x.exp import Exp
from src.models.twod.yolo.x.yolox_dataset import YOLOXDataset
import argparse


def main():
    server_ip = "10.0.0.1"

    test_client = SimpleNetworkClient(PickleMessageSerializer(), PickleMessageDeserializer())
    test_client.connect(server_ip)

    test_provider = NetworkFrameInstanceProvider(test_client)
    test_prefetcher = Prefetcher(test_provider, DatasetSplit.TEST, 8, fetch_timeout=60)
    test_prefetcher.run()

    test_set = YOLOXDataset(test_prefetcher)

    exp = Exp(train_set=None, val_set=None, test_set=test_set)

    args = argparse.Namespace(
        batch_size=8,
        devices=1,
        resume=False,
        start_epoch=None,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        experiment_name=exp.exp_name,
        ckpt="YOLOX_outputs/streaming_yolox/latest_ckpt.pth",
        fp16=False,
        fuse=False,
        cache=False,
        occupy=False,
        logger="tensorboard"
    )

    trainer = StreamingTrainer(exp, args)
    eval_loader = exp.get_eval_loader(batch_size=8, is_distributed=False, split="test")
    trainer.evaluate(eval_loader)

if __name__ == '__main__':
    main()