import traceback
import torch

from ext.yolox.yolox.evaluators import COCOEvaluator
from src.data.dataset.dataset_split import DatasetSplit
from src.data.streaming.prefetchers.batch_prefetcher import BatchPrefetcher
from src.models.twod.yolo.x.streaming_trainer import StreamingTrainer
from src.network.client.simple_network_client import SimpleNetworkClient
from src.network.messages.serialization.pickle_message_deserializer import PickleMessageDeserializer
from src.network.messages.serialization.pickle_message_serializer import PickleMessageSerializer
from src.network.network_frame_instance_provider import NetworkFrameInstanceProvider
from src.models.twod.yolo.x.exp import Exp
from src.models.twod.yolo.x.yolox_dataset import YOLOXDataset
from torch.utils.data import DataLoader
import argparse


def main():
    server_ip = "10.0.0.1"
    ckpt_path = "YOLOX_outputs/streaming_yolox/latest_ckpt.pth"

    client = SimpleNetworkClient(PickleMessageSerializer(), PickleMessageDeserializer())
    client.connect(server_ip)

    provider = NetworkFrameInstanceProvider(client)

    prefetcher = BatchPrefetcher(provider, DatasetSplit.TEST, 8, fetch_timeout=60)
    prefetcher.run()

    test_set = YOLOXDataset(prefetcher)
    test_loader = DataLoader(test_set, batch_size=None, num_workers=0, pin_memory=True, drop_last=False)

    try:
        exp = Exp(train_set=None, val_set=None)
        model = exp.get_model()
        model.eval()

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])

        evaluator = COCOEvaluator(
            dataloader=test_loader,
            img_size=exp.test_size,
            confthre=exp.test_conf,
            nmsthre=exp.nmsthre,
            num_classes=exp.num_classes,
            testdev=False
        )

        ap50_95, ap50, summary = evaluator.evaluate(model, distributed=False, half=False, return_outputs=False)
        print("✅ Evaluation summary:")
        print(summary)
    except Exception as e:
        print("\n❌ Exception caught during training:")
        traceback.print_exc()


if __name__ == '__main__':
    main()