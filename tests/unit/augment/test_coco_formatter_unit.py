import numpy as np
import pytest
from src.data.augment.coco_formatter import COCOFormatter

def test_coco_conversion():
    """Test conversion of batch data to COCO format."""
    formatter = COCOFormatter()
    sample_image = np.ones((224, 224, 3), dtype=np.uint8)
    sample_annotations = [(1, 50, 50, 100, 100)]
    batch_data = [(sample_image, sample_annotations)]

    coco_data = formatter.process(batch_data)

    assert "images" in coco_data
    assert "annotations" in coco_data
    assert len(coco_data["annotations"]) == 1
    assert coco_data["annotations"][0]["bbox"] == [50, 50, 100, 100]
