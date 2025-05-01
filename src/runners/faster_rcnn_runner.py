import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.factories.network_dataset_stream_factory import NetworkDatasetStreamFactory
from src.data.dataset.streams.providers.reusable_stream_provider import ReusableStreamProvider
from src.models.twod.rcnn.faster.streaming_dataset import StreamingDataset
from src.models.twod.rcnn.faster.trainer import Trainer

SERVER_IP = "10.0.0.1"


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    train_factory = NetworkDatasetStreamFactory(server_ip=SERVER_IP, split=DatasetSplit.TRAIN)
    train_provider = ReusableStreamProvider(train_factory.create_stream())
    dataset = StreamingDataset(train_provider, n_batches=400)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=5,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    trainer = Trainer(dataloader=dataloader, n_classes=5)
    trainer.train()


if __name__ == "__main__":
    main()
