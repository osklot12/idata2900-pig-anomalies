import math
import re
from typing import Callable, Any

import pytest

from src.data.dataset.splitters.consistent_dataset_splitter import ConsistentDatasetSplitter
from src.data.dataset.dataset_split import DatasetSplit


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


def _get_splits_before_and_after(splitter: ConsistentDatasetSplitter, func: Callable[..., Any]):
    """Returns the dataset splits before and after updating the dataset IDs."""
    splits_before = _get_splits(splitter)
    func()
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
def test_update_ids_adding_keep_consistency(dataset_ids):
    """Tests that adding IDs to the dataset keeps the existing distribution consistent."""
    # arrange
    splitter = ConsistentDatasetSplitter(dataset_ids=dataset_ids)
    updated_list = dataset_ids.copy()
    dataset_ids.add("id11")

    # act
    splits_before, splits_after = _get_splits_before_and_after(
        splitter,
        lambda: splitter.update_dataset(updated_list)
    )

    # assert
    for i in range(len(splits_before)):
        before_split = splits_before[i]
        after_split = splits_after[i]
        for id_ in before_split:
            assert id_ in after_split


@pytest.mark.unit
def test_update_ids_removal_keep_consistency(dataset_ids):
    """Tests that removing IDs from the dataset keeps the existing distribution consistent."""
    # arrange
    splitter = ConsistentDatasetSplitter(dataset_ids=dataset_ids)
    updated_list = dataset_ids.copy()
    id_to_remove = "id3"
    updated_list.remove(id_to_remove)

    # act
    splits_before, splits_after = _get_splits_before_and_after(
        splitter,
        lambda: splitter.update_dataset(updated_list)
    )

    # assert
    for i in range(len(splits_before)):
        before_split = splits_before[i]
        after_split = splits_after[i]

        for id_ in before_split:
            if id_ == id_to_remove:
                assert id_ not in after_split
            else:
                assert id_ in after_split


def test_add_instance_keeps_consistency():
    """Tests that add_instance() adds an instance while keeping consistency."""
    # arrange
    splitter = ConsistentDatasetSplitter()

    first_instance = "id1"
    second_instance = "id2"

    # act
    splitter.add_instance(first_instance)

    splits_before, splits_after = _get_splits_before_and_after(
        splitter,
        lambda: splitter.add_instance(second_instance)
    )

    # assert
    added_found = False
    for i in range(len(splits_before)):
        before_split = splits_before[i]
        after_split = splits_after[i]
        assert len(before_split) <= len(after_split)


        for id_ in before_split:
            assert id_ in after_split
            assert second_instance not in before_split
            added_found = second_instance in after_split

    assert added_found


def test_remove_instance_keeps_consistency():
    """Tests that remove_instance() removes an instance while keeping consistency."""
    # arrange
    splitter = ConsistentDatasetSplitter()

    first_instance = "id1"
    second_instance = "id2"

    splitter.add_instance(first_instance)
    splitter.add_instance(second_instance)

    # act
    splits_before, splits_after = _get_splits_before_and_after(
        splitter,
        lambda: splitter.remove_instance(first_instance)
    )

    # assert
    for i in range(len(splits_before)):
        before_split = splits_before[i]
        after_split = splits_after[i]
        print(f"Before: {before_split}")
        print(f"After: {after_split}")
        assert len(before_split) >= len(after_split)

        assert first_instance not in after_split

        for id_ in after_split:
            assert id_ in before_split