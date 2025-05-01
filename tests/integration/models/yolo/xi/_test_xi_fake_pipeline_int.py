from src.data.streaming.prefetchers.fake_batch_prefetcher import FakeBatchPrefetcher
from src.models.twod.yolo.xi.yoloxi_dataset import UltralyticsDataset
from src.models.twod.yolo.xi.yoloxi_eval_dataset import EvalUltralyticsDataset
from src.models.twod.yolo.xi.yoloxi_training_setup import TrainingSetup
import tempfile
import shutil
import pytest


@pytest.mark.integration
def test_yolov11_realistic_pipeline_with_empty_annotations():
    tmp_log_dir = tempfile.mkdtemp(prefix="tensorboard_test_real_")

    prefetcher = FakeBatchPrefetcher(batch_size=1)
    train_dataset = UltralyticsDataset(prefetcher, num_batches=2)
    eval_dataset = EvalUltralyticsDataset(prefetcher, num_batches=1)

    setup = TrainingSetup(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        epochs=2,
        log_dir=tmp_log_dir,
    )

    setup.train()

    metrics = setup.metrics
    print("📊 Realistic Eval Metrics:", metrics)

    assert metrics is not None, "Metrics should not be None even if some images have empty annotations"

    shutil.rmtree(tmp_log_dir)
