import torch
from unittest.mock import MagicMock
from src.models.twod.yolo.x.yolox_evaluator import YOLOXEvaluator
from src.schemas.schemas.signed_schema import SignedSchema
from src.schemas.schemas.metric_schema import MetricSchema
from src.schemas.brokers.schema_broker import SchemaBroker


class DummyYOLOXModel:
    def __init__(self):
        self.head = MagicMock()
        self.head.cls_preds = [torch.tensor([])]

    def eval(self):
        return self

    def __call__(self, images):
        return [{
            "bbox": torch.tensor([[10, 10, 50, 50]]),
            "scores": torch.tensor([0.9]),
            "cls": torch.tensor([1])
        } for _ in images]


class DummyYOLOXDataset:
    def __iter__(self):
        for _ in range(2):  # Simulate 2 batches
            yield {
                "img": torch.rand((2, 3, 640, 640)),  # Batch of 2 images
                "cls": torch.tensor([1, 1]),
                "bboxes": torch.tensor([[10, 10, 50, 50], [15, 15, 60, 60]]),
                "batch_idx": torch.tensor([0, 1]),
            }


def test_yolox_evaluator_logs_and_broadcasts_metrics(tmp_path):
    model = DummyYOLOXModel()
    dataset = DummyYOLOXDataset()
    broker = MagicMock(spec=SchemaBroker)

    evaluator = YOLOXEvaluator(model, dataset, broker, log_dir=str(tmp_path))
    metrics = evaluator.evaluate(num_batches=1)

    # ✅ Check keys exist
    assert "mAP@0.5" in metrics
    assert "Recall@100" in metrics
    assert "F1" in metrics

    # ✅ Check broker interaction
    broker.notify.assert_called_once()
    schema = broker.notify.call_args[0][0]
    assert isinstance(schema, SignedSchema)
    assert isinstance(schema.schema, MetricSchema)
