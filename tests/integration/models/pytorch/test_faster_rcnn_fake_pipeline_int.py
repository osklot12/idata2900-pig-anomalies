import tempfile
import shutil
import pytest

from src.models.twod.fastrcnn.train_fastrcnn import RCNNTrainer
from src.data.streaming.prefetchers.fake_batch_prefetcher import FakeBatchPrefetcher


@pytest.mark.integration
def test_rcnn_trains_and_evaluates_with_realistic_prefetcher():
    log_dir = tempfile.mkdtemp(prefix="tensorboard_test_rcnn_")

    prefetcher = FakeBatchPrefetcher(batch_size=4)
    trainer = RCNNTrainer(prefetcher=prefetcher)

    trainer.train()  # optionally change to trainer.train(max_iters=2) if you parameterize it
    trainer.evaluate(num_batches=2)

    print("📊 RCNN Evaluation Metrics:", trainer._last_metrics)
    assert trainer._last_metrics is not None

    shutil.rmtree(log_dir)