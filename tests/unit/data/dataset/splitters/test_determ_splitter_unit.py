import pytest

from src.data.dataset.splitters.determ_splitter import DetermSplitter


@pytest.fixture
def strings():
    """Fixture to provide a list of strings."""
    return [
        "string1", "string2", "string3",
        "string4", "string5", "string6",
        "string7", "string8", "string9"
    ]


@pytest.mark.unit
def test_split_all_into_first_split(strings):
    """Tests that setting the threshold to 1 will put every string into the first split."""
    # act
    splitter = DetermSplitter(strings, threshold=1)

    # assert
    assert len(splitter.first_split) == len(strings)
    assert len(splitter.second_split) == 0


@pytest.mark.unit
def test_split_all_into_second_split(strings):
    """Tests that setting the threshold to 0 will put every string into the second split."""
    # act
    splitter = DetermSplitter(strings, threshold=0)

    # assert
    assert len(splitter.first_split) == 0
    assert len(splitter.second_split) == len(strings)


@pytest.mark.unit
def test_split_determinism(strings):
    """Tests that the splits are created deterministically."""
    # arrange
    splitters = []

    # act
    for _ in range(100):
        splitters.append(DetermSplitter(strings))

    # assert
    first_splits = []
    second_splits = []
    for splitter in splitters:
        first_splits.append(splitter.first_split)
        second_splits.append(splitter.second_split)

    assert (
            len(set(tuple(s) for s in first_splits)) ==
            len(set(tuple(s) for s in second_splits)) ==
            1
    )
