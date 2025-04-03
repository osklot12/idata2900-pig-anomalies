import torch
import pytest
from enum import Enum
from unittest.mock import patch, MagicMock

from src.data.dataclasses.bbox import BBox
from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.models.twod.yolo.xi.train_yolo_eleven import YOLOv11Trainer
from src.schemas.metric_schema import MetricSchema
from src.data.dataset.dataset_split import DatasetSplit
from src.data.providers.batch_provider import BatchProvider
from src.data.streaming.prefetchers.batch_prefetcher import BatchPrefetcher


# -------------------------------
# Dummy enum class for testing
# -------------------------------

class DummyBehaviorClass(Enum):
    TAIL_BITING = 0
    BELLY_NOSING = 1
    EAR_BITING = 2
    TAIL_DOWN = 3


# -------------------------------
# Dummy BatchProvider & Prefetcher
# -------------------------------

class DummyBatchProvider(BatchProvider):
    def get_batch(self, split, batch_size):
        image = (torch.rand(640, 640, 3) * 255).byte().numpy()
        annotation = AnnotatedBBox(
            cls=DummyBehaviorClass.TAIL_BITING,
            bbox=BBox(x=100, y=100, width=50, height=50)
        )
        return [AnnotatedFrame(frame=image, annotations=[annotation])]


class FakePrefetcher(BatchPrefetcher):
    def __init__(self):
        super().__init__(
            batch_provider=DummyBatchProvider(),
            split=DatasetSplit.TRAIN,
            batch_size=1
        )

    def get(self):
        return self._provider.get_batch(self._split, self._batch_size)


# -------------------------------
# Integration Tests
# -------------------------------

@pytest.mark.integration
def test_yolov11_training_runs_one_epoch():
    """Ensures the YOLOv11 model can complete one training epoch."""
    trainer = YOLOv11Trainer(
        prefetcher=FakePrefetcher(),
        model_variant="yolo11m-obb.pt",
        epochs=1
    )

    result = trainer.train()

    assert result == "YOLOv11 training complete."


@pytest.mark.integration
@patch("src.schemas.training_metric_formater.TrainingMetricsCalculator")
def test_yolov11_metrics_are_formatted_and_sent(mock_calc_class):
    """Ensures metrics are formatted and pushed to the dashboard."""
    mock_metric = MagicMock()
    mock_metric.calculate.return_value = {
        "IoU": 0.85,
        "mAP": 0.75,
        "mAP@0.5": 0.9,
        "Recall@100": 0.8,
        "F1": 0.78
    }
    mock_calc_class.return_value = mock_metric

    mock_broker = MagicMock()

    trainer = YOLOv11Trainer(
        prefetcher=FakePrefetcher(),
        model_variant="yolo11m-obb.pt",
        epochs=1
    )

    trainer.train()

    # Ensure metrics_broker.notify is called
    assert mock_broker.notify.called, "Expected metrics_broker.notify to be called."

    # Ensure the schema contains MetricSchema
    schema_wrapper = mock_broker.notify.call_args[0][0]
    assert isinstance(schema_wrapper.schema, MetricSchema)

    # Ensure 'mAP' is present in the metrics
    assert "mAP" in schema_wrapper.schema.metrics

    # Ensure the correct issuer ID is passed
    assert schema_wrapper.issuer_id == "yolov11"
