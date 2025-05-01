import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.factories.network_dataset_stream_factory import NetworkDatasetStreamFactory
from src.data.dataset.streams.providers.reusable_stream_provider import ReusableStreamProvider
from src.models.twod.rcnn.faster.streaming_dataset import StreamingDataset

SERVER_IP = "10.0.0.1"

def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    device = torch.device("cuda" if torch.cude.is_available() else "cpu")
    print(f"Using device: {device}")

    train_factory = NetworkDatasetStreamFactory(server_ip=SERVER_IP, split=DatasetSplit.TRAIN)
    train_provider = ReusableStreamProvider(train_factory.create_stream())
    dataset = StreamingDataset(train_provider, n_batches=400)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=2,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    model = fasterrcnn_resnet50_fpn(num_classes=5)
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
    )

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train(epoch)
        total_loss = 0.0

        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch + 1}] Total loss: {total_loss:.4f}")

if __name__ == "__main__":
    main()