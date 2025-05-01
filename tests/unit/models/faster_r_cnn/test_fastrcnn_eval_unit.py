import pytest
import torch
from unittest.mock import MagicMock, patch

from src.models.twod.fastrcnn.train_fastrcnn import RCNNTrainer
from src.schemas.schemas.metric_schema import MetricSchema


@pytest.mark.unit
@patch("src.models.twod.fastrcnn.train_fastrcnn.metrics_broker.notify")
def test_evaluate_calculates_iou_and_emits_metrics(mock_notify):
    # --- Create mock prefetcher with fake frames ---
    mock_frame = MagicMock()
    mock_frame.frame = torch.zeros((224, 224, 3), dtype=torch.uint8)

    mock_ann = MagicMock()
    mock_ann.bbox = [50, 60, 100, 80]  # x, y, w, h
    mock_ann.category_id = 1
    mock_frame.annotations = [mock_ann]

    mock_prefetcher = MagicMock()
    mock_prefetcher.get.side_effect = [[mock_frame]] * 2  # two batches

    # --- Create trainer ---
    trainer = RCNNTrainer(prefetcher=mock_prefetcher)
    trainer.setup()

    # --- Patch model with dummy predictions matching the annotations ---
    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_model.side_effect = lambda imgs: [
        {
            "boxes": torch.tensor([[50, 60, 150, 140]], dtype=torch.float32),
            "labels": torch.tensor([1])
        }
        for _ in imgs
    ]
    trainer.model = mock_model

    # --- Patch tensor conversion to match model expectations ---
    trainer._convert_to_tensors = lambda batch: (
        [torch.rand(3, 224, 224)],
        [{"boxes": torch.tensor([[50, 60, 150, 140]]), "labels": torch.tensor([1])}]
    )

    # --- Run evaluation ---
    result = trainer.evaluate(num_batches=2)

    # --- Assertions ---
    assert result == "Evaluation completed."
    assert mock_notify.call_count == 1

    # Extract emitted metrics and check contents
    schema = mock_notify.call_args[0][0]
    metrics = schema.schema
    assert isinstance(metrics, MetricSchema)
    assert metrics.metrics["map50"] > 0
    assert metrics.metrics["recall"] > 0

