from unittest.mock import Mock

import numpy as np
import pytest
from _pytest._code import source

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataclasses.bbox import BBox
from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.dataset.virtual_dataset import VirtualDataset
from src.data.dataset_split import DatasetSplit
from tests.utils.test_annotation_label import TestAnnotationLabel


@pytest.fixture
def source_ids():
    """Fixture to provide a set of source IDs."""
    return {
        "id1", "id2", "id3", "id4", "id5", "id6"
    }


@pytest.fixture
def id_provider(source_ids):
    """Fixture to provide a mock SourceIdProvider instance."""
    provider = Mock()
    provider.get_source_ids.return_value = source_ids

    return provider


@pytest.fixture
def splitter(source_ids):
    """Fixture to provide a mock DatasetSplitter instance."""
    mock_splitter = Mock()

    train_ids = {"id1", "id2"}
    val_ids = {"id3", "id4"}
    test_ids = {"id5", "id6"}

    mock_splitter.get_split.side_effect = lambda split: {
        DatasetSplit.TRAIN: train_ids,
        DatasetSplit.VAL: val_ids,
        DatasetSplit.TEST: test_ids
    }[split]

    mock_splitter.get_split_for_id.side_effect = lambda id_: (
        DatasetSplit.TRAIN if id_ in train_ids else
        DatasetSplit.VAL if id_ in val_ids else
        DatasetSplit.TEST if id_ in test_ids else None
    )

    mock_splitter.get_split_ratio.side_effect = lambda split: {
        DatasetSplit.TRAIN: len(train_ids) / len(source_ids),
        DatasetSplit.VAL: len(val_ids) / len(source_ids),
        DatasetSplit.TEST: len(test_ids) / len(source_ids)
    }[split]

    return mock_splitter


@pytest.fixture
def standard_dataset(id_provider, splitter):
    """Fixture to provide a standard VirtualDataset instance."""
    return VirtualDataset(
        id_provider=id_provider,
        splitter=splitter,
        max_sources=10,
        max_frames_per_source=20
    )


def _gen_dummy_annotated_frames(n: int, source: str):
    """Helper function to generate dummy StreamedAnnotatedFrame instances."""
    anno_frames = []
    for i in range(n):
        anno_frames.append(
            StreamedAnnotatedFrame(
                source=source,
                index=i,
                frame=np.random.randint(0, 256, size=(240, 320, 3), dtype=np.uint8),
                annotations=[
                    AnnotatedBBox(
                        cls=TestAnnotationLabel.CODING,
                        bbox=BBox(0.6346, 0.4726, 0.1262, 0.0983)
                    )
                ],
                end_of_stream=False
            )
        )

    return anno_frames


@pytest.fixture
def fed_dataset(standard_dataset):
    """Fixture to provide a VirtualDataset with already existing data."""
    n_frames = 10
    frame_lists = [
        _gen_dummy_annotated_frames(n_frames, "id1"),
        _gen_dummy_annotated_frames(n_frames, "id2"),
        _gen_dummy_annotated_frames(n_frames, "id3"),
        _gen_dummy_annotated_frames(n_frames, "id4"),
        _gen_dummy_annotated_frames(n_frames, "id5"),
        _gen_dummy_annotated_frames(n_frames, "id6")
    ]

    for frame_list in frame_lists:
        for frame in frame_list:
            standard_dataset.feed(frame)

    return standard_dataset


def test_feed_valid_frames(standard_dataset):
    """Tests that feeding valid frames to VirtualDataset succeeds."""
    # arrange
    n_frames = 10
    frames = _gen_dummy_annotated_frames(n_frames, "id2")

    # act
    for frame in frames:
        standard_dataset.feed(frame)

    # assert
    assert standard_dataset.get_frame_count(DatasetSplit.TRAIN) == n_frames
    assert standard_dataset.get_frame_count(DatasetSplit.VAL) == 0
    assert standard_dataset.get_frame_count(DatasetSplit.TEST) == 0


def test_feed_invalid_frames(standard_dataset):
    """Tests that feeding invalid frames to VirtualDataset results in rejection."""
    # arrange
    n_frames = 10
    frames = _gen_dummy_annotated_frames(n_frames, "unknown-id")

    # act
    for frame in frames:
        standard_dataset.feed(frame)

    # assert
    assert standard_dataset.get_frame_count(DatasetSplit.TRAIN) == 0
    assert standard_dataset.get_frame_count(DatasetSplit.VAL) == 0
    assert standard_dataset.get_frame_count(DatasetSplit.TEST) == 0


def test_feed_frames_over_frame_capacity(standard_dataset):
    """Tests that feeding more frames than VirtualDataset's set capacity allows will evict older frames."""
    # arrange
    n_frames = 30
    frames = _gen_dummy_annotated_frames(n_frames, "id2")

    # act
    for frame in frames:
        standard_dataset.feed(frame)

    # assert
    assert standard_dataset.get_frame_count(DatasetSplit.TRAIN) == 20


def test_feed_frames_over_source_capacity(id_provider, splitter):
    """Tests that feeding frames from more distinct sources than allowed will evict older sources."""
    # arrange
    dataset = VirtualDataset(
        id_provider=id_provider,
        splitter=splitter,
        max_sources=3,
        max_frames_per_source=20
    )

    n_frames = 1
    frame_lists = [
        _gen_dummy_annotated_frames(n_frames, "id1"),
        _gen_dummy_annotated_frames(n_frames, "id2")
    ]

    # act
    for frame_list in frame_lists:
        for frame in frame_list:
            dataset.feed(frame)

    # assert
    assert dataset.get_frame_count(DatasetSplit.TRAIN) == 1


def test_get_shuffled_batch_returns_valid_frames(fed_dataset):
    """Tests that get_shuffled_batch() returns valid AnnotatedFrame instances."""
    # act
    batch = fed_dataset.get_shuffled_batch(split=DatasetSplit.TRAIN, batch_size=10)

    # assert
    for frame in batch:
        assert isinstance(frame, AnnotatedFrame)

def test_get_shuffled_batch_non_determinism(fed_dataset):
    """Tests that get_shuffled_batch() is non-deterministic."""
    # assert
    batches = []

    # act
    for _ in range(10):
        batches.append(fed_dataset.get_shuffled_batch(split=DatasetSplit.TEST, batch_size=10))

    # assert
    n_batches_eq = 0
    for i in range(len(batches)):
        j = (i + 1) % len(batches)
        while not j == i:
            n_frames_eq = 0
            for k in range(len(batches[i])):
                eq = np.array_equal(batches[i][k].frame, batches[j][k].frame)
                if eq:
                    n_frames_eq += 1

            if n_frames_eq == len(batches[i]):
                n_batches_eq += 1
            j = (j + 1) % len(batches)

    assert not n_batches_eq == len(batches)