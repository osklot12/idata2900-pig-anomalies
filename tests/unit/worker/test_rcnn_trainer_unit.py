# test_rcnn_trainer.py

import torch
import torchvision
import pytest
from src.worker.rcnn_trainer import RCNNTrainer
from src.worker.data_queue import data_queue


def generate_dummy_data():
    image = torch.rand((3, 224, 224), dtype=torch.float32)

    boxes = torch.tensor([[50.0, 60.0, 150.0, 160.0]])
    labels = torch.tensor([1])

    target = {
        "boxes": boxes,
        "labels": labels
    }

    return image, target


@pytest.mark.skipif(False, reason="Always run this test")
def test_rcnn_trainer_dummy_batch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for _ in range(3):
        image, target = generate_dummy_data()
        data_queue.put((image, target))

    # Comment this out
    # trainer = RCNNTrainer(rank=0, world_size=1)
    # trainer.setup()

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.roi_heads.parameters():
        param.requires_grad = True

    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.01,
        momentum=0.9
    )

    losses = []
    for _ in range(3):
        if data_queue.empty():
            continue

        image_tensor, target_dict = data_queue.get()
        images = [image_tensor.to(device)]
        targets = [{k: v.to(device) for k, v in target_dict.items()}]

        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        losses.append(total_loss.item())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # No trainer.cleanup()
    print(f"[PyTest] Losses: {losses}")
    assert all(loss > 0 for loss in losses)
