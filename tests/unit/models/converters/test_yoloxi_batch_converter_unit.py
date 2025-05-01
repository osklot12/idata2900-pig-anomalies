import numpy as np
import pytest
from enum import Enum, auto

import torch

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataclasses.bbox import BBox
from src.models.converters.xi.ultralytics_batch_converter import UltralyticsBatchConverter


class DummyClass(Enum):
    A = auto()
    B = auto()


@pytest.mark.unit
def test_converter_outputs_expected_format_for_ultralytics_obb():
    """Tests that the UltralyticsBatchConverter correctly formats OBB training samples."""

    # arrange
    img = np.ones((100, 200, 3), dtype=np.uint8) * 127

    annotations = [
        AnnotatedBBox(cls=DummyClass.A, bbox=BBox(10, 20, 40, 30)),
        AnnotatedBBox(cls=DummyClass.B, bbox=BBox(50, 60, 20, 10)),
    ]

    frame = AnnotatedFrame(frame=img, annotations=annotations)
    batch = [frame]

    # act
    output = UltralyticsBatchConverter.convert(batch=batch)[0]

    # assert
    assert "img" in output
    assert "instances" in output
    assert "bboxes" in output["instances"]
    assert "cls" in output["instances"]

    img_tensor = output["img"]
    bboxes = output["instances"]["bboxes"]
    classes = output["instances"]["cls"]

    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (100, 200, 3)
    assert img_tensor.dtype == torch.uint8

    assert isinstance(bboxes, torch.Tensor)
    assert bboxes.shape == (2, 5)  # (cx, cy, w, h, angle)

    expected_box_1 = [30.0, 35.0, 40.0, 30.0, 0.0]  # cx, cy, w, h, angle
    expected_box_2 = [60.0, 65.0, 20.0, 10.0, 0.0]

    np.testing.assert_allclose(bboxes[0].numpy(), expected_box_1, rtol=1e-2)
    np.testing.assert_allclose(bboxes[1].numpy(), expected_box_2, rtol=1e-2)

    assert classes.tolist() == [1, 2]
