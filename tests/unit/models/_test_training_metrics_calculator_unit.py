import torch
import pytest
from unittest.mock import MagicMock, patch
from src.models.training_metrics_calculator import TrainingMetricsCalculator

@pytest.mark.unit
def test_box_iou_exact_overlap():
    calc = TrainingMetricsCalculator()
    box = torch.tensor([0, 0, 10, 10], dtype=torch.float32)
    assert calc._box_iou(box, box) == 1.0

@pytest.mark.unit
def test_box_iou_partial_overlap():
    calc = TrainingMetricsCalculator()
    box1 = torch.tensor([0, 0, 10, 10], dtype=torch.float32)
    box2 = torch.tensor([5, 5, 15, 15], dtype=torch.float32)
    iou = calc._box_iou(box1, box2)
    assert 0.0 < iou < 1.0

@pytest.mark.unit
def test_box_iou_no_overlap():
    calc = TrainingMetricsCalculator()
    box1 = torch.tensor([0, 0, 10, 10], dtype=torch.float32)
    box2 = torch.tensor([20, 20, 30, 30], dtype=torch.float32)
    assert calc._box_iou(box1, box2) == 0.0

@pytest.mark.unit
def test_calculate_iou_single_match():
    calc = TrainingMetricsCalculator()
    predictions = [{"boxes": torch.tensor([[0, 0, 10, 10]])}]
    targets = [{"boxes": torch.tensor([[0, 0, 10, 10]])}]
    iou = calc._calculate_iou(predictions, targets)
    assert iou == 1.0

@pytest.mark.unit
@patch("src.models.training_metrics_calculator.MeanAveragePrecision")
def test_calculate_map_mocked(mock_map_class):
    mock_metric = MagicMock()
    mock_metric.compute.return_value = {
        "map": 0.42,
        "map_50": 0.7,
        "mar_100": 0.65
    }
    mock_map_class.return_value = mock_metric

    calc = TrainingMetricsCalculator()
    preds = [{"boxes": torch.empty((0, 4)), "labels": torch.empty(0), "scores": torch.empty(0)}]
    targs = [{"boxes": torch.empty((0, 4)), "labels": torch.empty(0)}]
    result = calc._calculate_map(preds, targs)

    assert result["mAP"] == 0.42
    assert result["mAP@0.5"] == 0.7
    assert result["Recall@100"] == 0.65
    assert 0 <= result["F1"] <= 1

@pytest.mark.unit
@patch("src.models.training_metrics_calculator.MeanAveragePrecision")
def test_calculate_full_mocked(mock_map_class):
    mock_metric = MagicMock()
    mock_metric.compute.return_value = {
        "map": 0.5,
        "map_50": 0.8,
        "mar_100": 0.6
    }
    mock_map_class.return_value = mock_metric

    calc = TrainingMetricsCalculator()
    preds = [{"boxes": torch.tensor([[0, 0, 10, 10]]), "labels": torch.tensor([1]), "scores": torch.tensor([0.9])}]
    targs = [{"boxes": torch.tensor([[0, 0, 10, 10]]), "labels": torch.tensor([1])}]
    result = calc.calculate(preds, targs)

    assert "IoU" in result
    assert "mAP" in result
    assert "F1" in result

