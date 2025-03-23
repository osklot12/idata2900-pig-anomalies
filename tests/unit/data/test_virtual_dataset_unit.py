import threading

import numpy as np
import pytest

from src.data.dataset_split import DatasetSplit
from src.data.dataset.virtual_dataset import VirtualDataset


@pytest.fixture
def source_ids():
    """Fixture to provide dataset source ids."""
    return [
        "video1", "video2", "video3",
        "video4", "video5", "video6",
        "video7", "video8", "video9"
    ]

@pytest.fixture
def virtual_dataset(source_ids):
    """Creates a VirtualDataset instance with mock dataset IDs."""
    return VirtualDataset(
        dataset_ids=source_ids, max_sources=10, max_frames_per_source=15,
        train_ratio=0.4, val_ratio=0.3, test_ratio=0.3
    )

@pytest.fixture
def sample_frame():
    """Returns a sample frame."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)

@pytest.fixture
def sample_annotation():
    """Returns a sample annotation."""
    return [("label", 100, 100, 50, 50)]

def test_dataset_initialization(virtual_dataset):
    """Tests if VirtualDataset correctly assigns train, validation, and test splits."""
    # arrange
    dataset_count = len(virtual_dataset.dataset_ids)

    # assert
    assert len(virtual_dataset.train_ids) + len(virtual_dataset.val_ids) + len(
        virtual_dataset.test_ids) == dataset_count
    assert set(virtual_dataset.train_ids + virtual_dataset.val_ids + virtual_dataset.test_ids) == set(
        virtual_dataset.dataset_ids)


def test_feed_and_retrieve_frames(virtual_dataset, sample_frame, sample_annotation):
    """Tests feeding frames into the dataset and retrieving them."""
    # arrange
    source = "video1"

    # act
    for i in range(10):
        virtual_dataset.feed(source, i, sample_frame, sample_annotation, end_of_stream=False)

    # assert
    buffer = virtual_dataset._get_buffer_for_source(source)

    source_buffer = buffer.at(source)

    assert source_buffer is not None, "Source buffer should exist."
    assert source_buffer.size() == 10, "Should contain 10 frames."

    retrieved_frame, retrieved_annotation = source_buffer.at(5)
    assert np.array_equal(retrieved_frame, sample_frame), "Frame data should match."
    assert retrieved_annotation == sample_annotation, "Annotations should match."


def test_get_frame_count(virtual_dataset, sample_frame, sample_annotation):
    """Tests counting the number of stored frames in each split."""
    # arrange
    source_train = virtual_dataset.train_ids[0]
    source_val = virtual_dataset.val_ids[0]

    for i in range(10):
        virtual_dataset.feed(source_train, i, sample_frame, sample_annotation, end_of_stream=False)

    for i in range(5):
        virtual_dataset.feed(source_val, i, sample_frame, sample_annotation, end_of_stream=False)

    # act
    count_train = virtual_dataset.get_frame_count(DatasetSplit.TRAIN)
    count_val = virtual_dataset.get_frame_count(DatasetSplit.VAL)
    count_test = virtual_dataset.get_frame_count(DatasetSplit.TEST)

    # assert
    assert count_train == 10, "Train split should contain 10 frames."
    assert count_val == 5, "Validation split should contain 5 frames."
    assert count_test == 0, "Test split should contain 10 frames."


def test_get_shuffled_batch(virtual_dataset, sample_frame, sample_annotation):
    """Tests the randomized batch sampling method."""
    # arrange
    source = virtual_dataset.train_ids[0]

    for i in range(30):
        virtual_dataset.feed(source, i, sample_frame, sample_annotation, end_of_stream=False)

    # act
    batch_size = 10
    batch = virtual_dataset.get_shuffled_batch(DatasetSplit.TRAIN, batch_size)

    # assert
    assert len(batch) == batch_size, f"Batch should contain {batch_size} frames."
    assert all(isinstance(item, tuple) and len(item) == 2 for item in batch), \
        "Batch items should be (frame, annotation tuples)."

def test_get_shuffled_batch_fails_on_empty_split(virtual_dataset):
    """Ensures batch retrieval fails when no data is available."""
    split = DatasetSplit.TRAIN
    batch_size = 10
    with pytest.raises(ValueError, match=f"Not enough available data in split {split.name} for batch size {batch_size}."):
        virtual_dataset.get_shuffled_batch(DatasetSplit.TRAIN, batch_size)

def test_get_shuffled_batch_is_non_deterministic(virtual_dataset, sample_annotation):
    """
    Ensures that batch retrieval is not always returning the same frames in the same order.
    Due to the randomness involved, this test may give a false negative, so run it a couple of times if needed.
    """
    # arrange
    source = virtual_dataset.train_ids[0]

    n_frames = 1000
    batch_size = 10
    n_batches = 5
    retrieved_batches = []

    for i in range(n_frames):
        unique_frame = np.full((1080, 1920, 3), i % 256, dtype=np.uint8)
        virtual_dataset.feed(source, i, unique_frame, sample_annotation, end_of_stream=False)

    # act
    for _ in range(n_batches):
        batch = virtual_dataset.get_shuffled_batch(DatasetSplit.TRAIN, batch_size)
        retrieved_batches.append(batch)

    # assert
    batch_sets = [set(np.mean(frame) for frame, _ in batch) for batch in retrieved_batches]

    unique_batches = len(set(tuple(sorted(batch)) for batch in batch_sets))

    assert unique_batches > 1, "Shuffled batches should not all be identical."



def test_thread_safety(virtual_dataset, sample_frame, sample_annotation):
    """Tests that feeding and batch retrieval can run safely in multiple threads."""
    # arrange
    source = virtual_dataset.train_ids[0]

    def feed_data():
        for i in range(15):
            virtual_dataset.feed(source, i, sample_frame, sample_annotation, end_of_stream=False)

    def retrieve_batches():
        for _ in range(3):
            try:
                virtual_dataset.get_shuffled_batch(DatasetSplit.TRAIN, batch_size=5)
            except ValueError:
                pass

    feed_thread = threading.Thread(target=feed_data)
    batch_thread = threading.Thread(target=retrieve_batches)

    # act
    feed_thread.start()
    batch_thread.start()

    feed_thread.join()
    batch_thread.join()

    # assert
    assert virtual_dataset.get_frame_count(DatasetSplit.TRAIN) == 15, "Train split should contain 15 frames."

def test_max_buffer_eviction(virtual_dataset, sample_frame, sample_annotation):
    """Ensures that the dataset respects max_frames_per_source by evicting old frames."""
    # arrange
    source = virtual_dataset.train_ids[0]

    # act
    for i in range(virtual_dataset.max_frames_per_source + 5):
        virtual_dataset.feed(source, i, sample_frame, sample_annotation, end_of_stream=False)

    # assert
    assert virtual_dataset.get_frame_count(DatasetSplit.TRAIN) == virtual_dataset.max_frames_per_source, \
        f"Dataset should retain only {virtual_dataset.max_frames_per_source} frames, but got {virtual_dataset.get_frame_count(DatasetSplit.TRAIN)}."

    buffer = virtual_dataset._get_buffer_for_source(source)
    source_buffer = buffer.at(source)

    assert source_buffer is not None, "Source buffer should exist."
    assert source_buffer.has(4) is False, "Oldest frames should have been evicted."
    assert source_buffer.has(virtual_dataset.max_frames_per_source + 4) is True, "Newest frames should retained."