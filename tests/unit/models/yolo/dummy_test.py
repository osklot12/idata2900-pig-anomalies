import torch
import pytest

from src.data.dataset.dataset_split import DatasetSplit
from src.data.providers.batch_provider import BatchProvider
from src.data.streaming.prefetchers.batch_prefetcher import BatchPrefetcher
from src.models.twod.yolo.train_yolo_eleven import YOLOv11Trainer


# ---------------------------
# Dummy components for isolation
# ---------------------------

class DummyBatchProvider(BatchProvider):
    def get_batch(self, split, batch_size):
        return []


class DummyPrefetcher(BatchPrefetcher):
    def __init__(self):
        super().__init__(
            batch_provider=DummyBatchProvider(),
            split=DatasetSplit.TRAIN,
            batch_size=1
        )

    def get(self):
        return []


# ---------------------------
# Unit Test
# ---------------------------

@pytest.mark.unit
def test_yolov11_model_forward_pass():
    trainer = YOLOv11Trainer(
        prefetcher=DummyPrefetcher(),
        model_variant="yolo11m-obb.pt",  # must be -obb
        epochs=1
    )
    trainer.setup()

    dummy_image = torch.rand(1, 3, 640, 640).to(trainer.device)

    with torch.no_grad():
        results = trainer.yolo.predict(dummy_image, verbose=False)

    assert isinstance(results, list)
    assert hasattr(results[0], "obb")
    assert results[0].obb is not None
    assert results[0].obb.xyxyxyxy is not None
