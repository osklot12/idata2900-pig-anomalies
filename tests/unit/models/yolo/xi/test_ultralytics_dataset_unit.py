import numpy as np
import pytest
from enum import Enum, auto

import torch

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataclasses.bbox import BBox
from src.models.twod.yolo.xi.yoloxi_dataset import UltralyticsDataset


class DummyClass(Enum):
    A = auto()
    B = auto()


class DummyPrefetcher:
    """Mock prefetcher that returns one batch repeatedly."""
    def get(self):
        img = np.ones((80, 120, 3), dtype=np.uint8) * 100
        annotations = [
            AnnotatedBBox(cls=DummyClass.A, bbox=BBox(10, 20, 40, 30)),
            AnnotatedBBox(cls=DummyClass.B, bbox=BBox(50, 60, 20, 10)),
        ]
        return [AnnotatedFrame(frame=img, annotations=annotations)]


@pytest.mark.unit
def test_ultralytics_dataset_iter_yields_correct_format():
    """Tests that UltralyticsDataset yields properly converted samples from the prefetcher."""

    # arrange
    dataset = UltralyticsDataset(prefetcher=DummyPrefetcher())
    data_iter = iter(dataset)

    # act
    sample = next(data_iter)

    # assert
    assert isinstance(sample, dict)
    assert "img" in sample
    assert "instances" in sample

    img = sample["img"]
    bboxes = sample["instances"]["bboxes"]
    classes = sample["instances"]["cls"]

    assert isinstance(img, torch.Tensor)
    assert img.shape == (80, 120, 3)
    assert img.dtype == torch.uint8

    assert bboxes.shape == (2, 5)  # cx, cy, w, h, angle
    np.testing.assert_allclose(bboxes[0].numpy(), [30.0, 35.0, 40.0, 30.0, 0.0], rtol=1e-2)
    np.testing.assert_allclose(bboxes[1].numpy(), [60.0, 65.0, 20.0, 10.0, 0.0], rtol=1e-2)

    assert classes.tolist() == [1, 2]
