import pytest

from src.data.dataset.selectors.determ_string_selector import DetermStringSelector


@pytest.fixture
def files():
    """Fixture to provide a list of files for testing."""
    return [
        "testfile1.mp4",
        "testfile2.mp4",
        "testfile3.mp4",
        "testfile4.mp4"
    ]


@pytest.mark.unit
def test_select_file_returns_none_on_full_cycle(files):
    """Tests that select_file returns None when all files have been selected."""
    # arrange
    selector = DetermStringSelector(files)
    initial_selections = []

    # act
    for _ in range(4):
        initial_selections.append(selector.next())

    last_selection = selector.next()

    # assert
    for file in initial_selections:
        assert file is not None

    assert last_selection is None


@pytest.mark.unit
def test_files_are_selected_once(files):
    """Tests that each files is only selected once."""
    # arrange
    selector = DetermStringSelector(files)
    selections = []

    # act
    for _ in range(4):
        selections.append(selector.next())

    # assert
    assert len(set(selections)) == len(selections)


@pytest.mark.unit
def test_seed_ensures_reproducibility(files):
    """Tests that setting the seed ensures reproducibility in the order of selectors."""
    # arrange
    selector_one = DetermStringSelector(files, seed=1)
    selector_two = DetermStringSelector(files, seed=1)
    first_selections = []
    second_selections = []

    # act
    for _ in range(4):
        first_selections.append(selector_one.next())
        second_selections.append(selector_two.next())

    # assert
    assert first_selections == second_selections
