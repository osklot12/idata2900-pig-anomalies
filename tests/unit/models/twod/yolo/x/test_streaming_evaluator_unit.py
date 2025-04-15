import numpy as np
import pytest
import torch
from torch.utils.data import IterableDataset

from src.models.twod.yolo.x.streaming_evaluator import StreamingEvaluator
from torch.utils.data import DataLoader


@pytest.fixture
def targets():
    """Fixture to provide dummy targets."""
    return torch.tensor([
        [[0, 10, 10, 10, 10],  # cls, cx, cy, w, h
         [1, 30, 30, 10, 10]]
    ])  # shape: (1, 2, 5)


@pytest.fixture
def outputs():
    """Fixture to provide dummy outputs."""
    return [torch.tensor([
        [10, 10, 10, 10, 0.9, 0],  # matches first GT (IoU = 1.0)
        [30, 30, 10, 10, 0.8, 1],  # matches second GT (IoU = 1.0)
    ])]


class DummyDataset(IterableDataset):
    """Dummy iterable dataset for testing."""

    def __init__(self, targets):
        self.targets = targets

    def __iter__(self):
        yield {
            "image": torch.rand(1, 3, 64, 64),
            "target": self.targets
        }

    def __len__(self):
        return 1


class DummyModel(torch.nn.Module):

    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs

    """Dummy model for testing."""

    def forward(self, x):
        return self.outputs


@pytest.fixture
def dataset(targets):
    """Fixture to provide a DummyDataset instance."""
    return DummyDataset(targets=targets)


@pytest.fixture
def dataloader(dataset):
    """Fixture to provide a dummy dataloader instance."""
    return DataLoader(dataset, batch_size=None)


@pytest.fixture
def model(outputs):
    """Fixture to provide a DummyModel instance"""
    return DummyModel(outputs=outputs)


@pytest.mark.unit
def test_streaming_evaluator_accurate_predictions(model, dataloader):
    """Tests that accurate predictions give correct metrics."""
    # arrange
    evaluator = StreamingEvaluator(
        model=model,
        dataloader=dataloader,
        device=torch.device("cpu"),
        num_classes=2,
        iou_thresh=0.5
    )

    # act
    results = evaluator.evaluate()

    # assert
    print("\n--- Mixed predictions test ---")
    for k, v in results.items():
        print(f"{k}: {v}")

    assert np.isclose(results["precision"], 1.0)
    assert np.isclose(results["recall"], 1.0)
    assert np.isclose(results["f1"], 1.0)
    assert np.isclose(results["mAP"], 1.0)
    assert np.allclose(results["confusion_matrix"][:2, :2], np.eye(2))


@pytest.mark.unit
def test_streaming_evaluator_mixed_predictions():
    """Tests that mixed predictions give correct metrics."""
    # arrange
    targets = torch.tensor([
        [[0, 10, 10, 10, 10],
         [1, 30, 30, 10, 10]]
    ])

    outputs = [torch.tensor([
        [10, 10, 10, 10, 0.9, 0],
        [30, 30, 10, 10, 0.8, 1],
        [50, 50, 10, 10, 0.7, 2],
    ])]

    dataset = DummyDataset(targets=targets)
    dataloader = DataLoader(dataset, batch_size=None)
    model = DummyModel(outputs=outputs)

    evaluator = StreamingEvaluator(
        model=model,
        dataloader=dataloader,
        device=torch.device("cpu"),
        num_classes=3,
        iou_thresh=0.5
    )

    # act
    results = evaluator.evaluate()

    print("\n--- Mixed predictions test ---")
    for k, v in results.items():
        print(f"{k}: {v}")

    assert results["precision"] < 1.0
    assert results["recall"] < 1.0
    assert results["mAP"] < 1.0
    assert results["confusion_matrix"][0][0] == 1
