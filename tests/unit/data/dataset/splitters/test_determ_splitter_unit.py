from typing import Tuple, List

import pytest

from src.data.dataset.splitters.determ_splitter import DetermSplitter


@pytest.fixture
def strings():
    """Fixture to provide a list of strings."""
    return [f"string_{i}" for i in range(1000)]


@pytest.mark.unit
def test_strings_provided_on_construction(strings):
    """Tests that passing the strings on construction makes valid splits."""
    # arrange
    splitter = DetermSplitter(strings)

    # act
    splits = splitter.splits

    # assert
    assert sum(len(split) for split in splits) == len(strings)

    flat_splits = [item for split in splits for item in split]
    for string in strings:
        assert string in flat_splits


@pytest.mark.unit
def test_giving_n_weights_produces_n_splits(strings):
    """Tests that giving n valid weights will produce n valid splits."""
    # arrange
    splitter = DetermSplitter(strings=strings, weights=[0.2 for _ in range(5)])

    # act
    splits = splitter.splits

    # assert
    assert len(splits) == 5
    for split in splits:
        assert len(split) > 0  # this could statistically fail, but the chances are astronomically low


@pytest.mark.unit
def test_adding_incrementally_gives_the_same_splits(strings):
    """Tests that adding strings incrementally instead of giving all in the constructor produces the same splits."""
    # arrange
    splitter = DetermSplitter()
    prefilled_splitter = DetermSplitter(strings=strings)

    # act
    for s in strings:
        splitter.add(s)

    # assert
    assert splitter.splits == prefilled_splitter.splits


@pytest.mark.unit
def test_giving_less_than_two_weights_raises():
    """Tests that giving less than two weights on construction causes a raise."""
    # act & assert
    with pytest.raises(ValueError):
        DetermSplitter(weights=[1])


@pytest.mark.unit
@pytest.mark.parametrize(
    "weights",
    [
        [0.2, 0.2, 0.2],
        [0.1, 0.5, 0.5],
        [1.0, 20.0, 200.0, 300.0]
    ]
)
def test_weights_not_summing_to_one_raises(weights):
    """Tests that giving weights that does not sum to one on construction causes a raise."""
    # act & assert
    with pytest.raises(ValueError):
        DetermSplitter(weights=weights)


@pytest.mark.unit
@pytest.mark.parametrize(
    "seed", [42, 35, 69, 140]
)
def test_seed_gives_determinism_in_splits(strings, seed):
    """Tests that the seed gives determinism in the splitting."""
    # act
    splitters = [DetermSplitter(strings=strings, seed=seed) for _ in range(10)]

    # assert
    split_structures = {
        tuple(tuple(split) for split in splitter.splits)
        for splitter in splitters
    }
    assert len(split_structures) == 1
