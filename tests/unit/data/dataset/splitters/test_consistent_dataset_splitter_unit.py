import math
import random
import re
from typing import List, Tuple, Iterable

import pytest

from src.data.dataset.splitters.consistent_dataset_splitter import ConsistentDatasetSplitter
from src.data.dataset_split import DatasetSplit


@pytest.fixture
def dataset_ids():
    """Fixture to provide the dataset IDs."""
    return {
        "id1", "id2", "id3", "id4", "id5",
        "id6", "id7", "id8", "id9", "id10"
    }


def _get_splits(splitter: ConsistentDatasetSplitter):
    """Returns all dataset splits for the splitter."""
    return (
        splitter.get_split(DatasetSplit.TRAIN),
        splitter.get_split(DatasetSplit.VAL),
        splitter.get_split(DatasetSplit.TEST)
    )


def _get_splits_before_and_after_update(splitter: ConsistentDatasetSplitter, updated_dataset_ids: Iterable[str]):
    """Returns the dataset splits before and after updating the dataset IDs."""
    splits_before = _get_splits(splitter)
    splitter.update_dataset(updated_dataset_ids)
    splits_after = _get_splits(splitter)

    return splits_before, splits_after


@pytest.mark.unit
def test_initialization_invalid_ratios():
    """Tests that initialization with invalid split ratios will raise an error."""
    # act & assert
    with pytest.raises(ValueError,
                       match=re.escape("The sum of train_ratio and val_ratio must be in the interval [0, 1]")):
        ConsistentDatasetSplitter(train_ratio=1, val_ratio=1)


@pytest.mark.unit
def test_get_split_ratio():
    """Tests that get_split_ratio() returns the correct split ratios."""
    # arrange
    ratios = (
        0.6,  # train
        0.2,  # val
        0.2  # test
    )

    splitter = ConsistentDatasetSplitter(train_ratio=ratios[0], val_ratio=ratios[1])

    # act
    returned_ratios = (
        splitter.get_split_ratio(DatasetSplit.TRAIN),
        splitter.get_split_ratio(DatasetSplit.VAL),
        splitter.get_split_ratio(DatasetSplit.TEST)
    )

    # assert
    for i in range(len(ratios)):
        assert math.isclose(returned_ratios[i], ratios[i], rel_tol=1e-9)


@pytest.mark.unit
def test_get_split_for_id(dataset_ids):
    """Tests that get_split_for_id() returns the correct split for the given id."""
    # arrange
    splitter = ConsistentDatasetSplitter(dataset_ids=dataset_ids, train_ratio=1, val_ratio=0)

    # act
    returned_split = splitter.get_split_for_id("id5")

    # assert
    assert returned_split == DatasetSplit.TRAIN


@pytest.mark.unit
def test_update_dataset_and_get_split(dataset_ids):
    """Tests that updating the dataset and getting the splits works as expected."""
    # arrange
    splitter = ConsistentDatasetSplitter()

    # act
    splitter.update_dataset(dataset_ids)
    train_split, val_split, test_split = _get_splits(splitter)

    # assert
    assert len(train_split) + len(val_split) + len(test_split) == len(dataset_ids)


@pytest.mark.unit
def test_adding_ids_keep_consistency(dataset_ids):
    """Tests that adding IDs to the dataset keeps the existing distribution consistent."""
    # arrange
    splitter = ConsistentDatasetSplitter(dataset_ids=dataset_ids)
    updated_list = dataset_ids.copy()
    dataset_ids.add("id11")

    # act
    splits_before, splits_after = _get_splits_before_and_after_update(splitter, updated_list)

    # assert
    for i in range(len(splits_before)):
        before_split = splits_before[i]
        after_split = splits_after[i]
        for id_ in before_split:
            assert id_ in after_split


@pytest.mark.unit
def test_removing_ids_keep_consistency(dataset_ids):
    """Tests that removing IDs from the dataset keeps the existing distribution consistent."""
    # arrange
    splitter = ConsistentDatasetSplitter(dataset_ids=dataset_ids)
    updated_list = dataset_ids.copy()
    id_to_remove = "id3"
    updated_list.remove(id_to_remove)

    # act
    splits_before, splits_after = _get_splits_before_and_after_update(splitter, updated_list)

    # assert
    for i in range(len(splits_before)):
        before_split = splits_before[i]
        print(f"Before split: {before_split}")
        after_split = splits_after[i]
        print(f"After split: {after_split}")
        for id_ in before_split:
            if id_ == id_to_remove:
                assert id_ not in after_split
            else:
                assert id_ in after_split
