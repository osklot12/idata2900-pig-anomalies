import time
from unittest.mock import MagicMock

import pytest

from src.data.dataset.dataset_split import DatasetSplit
from src.data.streaming.prefetchers.batch_prefetcher import BatchPrefetcher


@pytest.fixture
def batch_provider():
    """Fixture to provide a BatchProvider instance."""
    mock = MagicMock()
    mock.get_batch.return_value = ["dummy_batch"]
    return mock


@pytest.mark.unit
def test_prefetch_and_retrieve(batch_provider):
    """Tests that prefetching and retrieving batches works as expected."""
    # arrange
    prefetcher = BatchPrefetcher[str](
        batch_provider=batch_provider,
        split=DatasetSplit.TRAIN,
        batch_size=1
    )

    # act
    prefetcher.run()

    time.sleep(.1)

    batch = prefetcher.get()

    # assert
    assert batch == ["dummy_batch"]
    assert batch_provider.get_batch.call_count >= 1

    prefetcher.stop()


@pytest.mark.unit
def test_double_run_raises(batch_provider):
    """Tests that running the prefetcher while already running will raise."""
    # arrange
    prefetcher = BatchPrefetcher[str](
        batch_provider=batch_provider,
        split=DatasetSplit.TRAIN,
        batch_size=1
    )

    prefetcher.run()

    # act & assert
    with pytest.raises(RuntimeError, match="Prefetcher is already running"):
        prefetcher.run()

    prefetcher.stop()


@pytest.mark.unit
def test_stop_works_cleanly(batch_provider):
    """Tests that stopping the prefetcher works as expected."""
    # arrange
    prefetcher = BatchPrefetcher[str](
        batch_provider=batch_provider,
        split=DatasetSplit.TRAIN,
        batch_size=1
    )

    prefetcher.run()
    time.sleep(.1)

    # act
    prefetcher.stop()

    # assert
    assert prefetcher._thread is None
    assert not prefetcher._running.get()