import numpy as np
import pytest
from enum import Enum, auto

import torch

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataclasses.bbox import BBox
from src.models.converters.yolox_batch_converter import YOLOXBatchConverter


class DummyClass(Enum):
    A = auto()
    B = auto()


@pytest.mark.unit
def test_converter_outputs_expected_shapes_and_values():
    """Tests that the YOLOXBatchConverter correctly converts a batch into expected shapes and values."""
    # arrange
    img = np.ones((100, 200, 3), dtype=np.uint8) * 127

    annotations = [
        AnnotatedBBox(cls=DummyClass.A, bbox=BBox(.1, .2, .3, .4)),
        AnnotatedBBox(cls=DummyClass.B, bbox=BBox(.5, .6, .2, .1))
    ]

    frame = AnnotatedFrame(frame=img, annotations=annotations)
    batch = [frame]

    # act
    output = YOLOXBatchConverter.convert(batch=batch)

    # assert
    assert "img" in output
    assert "gt_boxes" in output
    assert "gt_classes" in output
    assert "gt_num" in output

    assert isinstance(output["img"], torch.Tensor)
    assert output["img"].shape == (1, 100, 200, 3)
    assert output["img"].dtype == torch.uint8

    gt_boxes = output["gt_boxes"][0]
    assert isinstance(gt_boxes, torch.Tensor)
    assert gt_boxes.shape == (2, 4)

    expected_box_1 = [20.0, 20.0, 80.0, 60.0]
    expected_box_2 = [100.0, 60.0, 140.0, 70.0]

    np.testing.assert_allclose(gt_boxes[0].numpy(), expected_box_1, rtol=1e-2)
    np.testing.assert_allclose(gt_boxes[1].numpy(), expected_box_2, rtol=1e-2)

    assert output["gt_classes"][0].tolist() == [1, 2]
    assert output["gt_num"] == [2]