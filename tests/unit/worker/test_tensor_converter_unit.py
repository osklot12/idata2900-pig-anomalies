# tests/test_tensor_converter.py
import numpy as np
from src.worker.tensor_converter import convert_to_tensor_format

def test_tensor_converter_basic():
    frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    annotations = [
        {"bbox": [10, 20, 30, 40], "label": 1},
        {"bbox": [50, 60, 70, 80], "label": 2}
    ]

    image_tensor, target = convert_to_tensor_format(frame, annotations)

    assert image_tensor.shape == (3, 224, 224)
    assert target["boxes"].shape == (2, 4)
    assert target["labels"].shape == (2,)
