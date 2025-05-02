import numbers

import numpy as np
import pytest
from src.utils.calculators.metrics.map_calculator import MAPCalculator


@pytest.mark.unit
def test_creation():
    calculator = MAPCalculator(num_classes=4, iou_threshold=0.5)
    assert calculator.num_classes == 4
    assert calculator.iou_threshold == 0.5
    assert hasattr(calculator.metric, "add")

@pytest.mark.unit
def test_update_adds_data_successfully():
    calculator = MAPCalculator(num_classes=2)

    # Simple valid prediction and ground truth
    gt = np.array([[10, 10, 50, 50, 1]], dtype=np.float32)
    pred = np.array([[10, 10, 50, 50, 0.9, 1]], dtype=np.float32)

    # Should not raise
    calculator.update(pred, gt)

    result = calculator.compute()
    assert isinstance(result, dict)
    assert "mAP" in result

@pytest.mark.unit
def test_map_computes_with_valid_data():
    calculator = MAPCalculator(num_classes=2, iou_threshold=0.5)

    gt = []
    pred = []

    for i in range(20):
        x1, y1 = i * 10, i * 10
        x2, y2 = x1 + 40, y1 + 40
        cls = 1

        gt.append([x1, y1, x2, y2, cls])
        pred.append([x1, y1, x2, y2, 0.9, cls])
        pred.append([x1 + 10, y1 + 10, x2 + 10, y2 + 10, 0.6, cls])

    gt = np.array(gt, dtype=np.float32)
    pred = np.array(pred, dtype=np.float32)

    calculator.update(pred, gt)
    result = calculator.compute()

    assert "mAP" in result
    assert isinstance(result["mAP"], numbers.Number)
    assert result["mAP"] >= 0.0

@pytest.mark.unit
def test_reset():
    calculator = MAPCalculator(num_classes=1)

    # Add sample data
    gt = np.array([[0, 0, 10, 10, 0]], dtype=np.float32)
    pred = np.array([[0, 0, 10, 10, 1.0, 0]], dtype=np.float32)
    calculator.update(pred, gt)

    # Reset
    calculator.reset()

    # Compute again: should return 0s
    result = calculator.compute()
    assert result["mAP"] == 0.0
    assert all(v == 0.0 for v in result["per_class_ap"].values())
