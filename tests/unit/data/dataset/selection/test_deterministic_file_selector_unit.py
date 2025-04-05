import pytest

from src.data.dataset.selection.deterministic_file_selector import DeterministicFileSelector


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
    selector = DeterministicFileSelector(files)
    initial_selections = []

    # act
    for _ in range(4):
        initial_selections.append(selector.select_file())

    last_selection = selector.select_file()

    # assert
    for file in initial_selections:
        assert file is not None

    assert last_selection is None


@pytest.mark.unit
def test_files_are_selected_once(files):
    """Tests that each files is only selected once."""
    # arrange
    selector = DeterministicFileSelector(files)
    selections = []

    # act
    for _ in range(4):
        selections.append(selector.select_file())

    # assert
    assert len(set(selections)) == len(selections)


@pytest.mark.unit
def test_seed_ensures_reproducibility(files):
    """Tests that setting the seed ensures reproducibility in the order of selection."""
    # arrange
    selector_one = DeterministicFileSelector(files, seed=1)
    selector_two = DeterministicFileSelector(files, seed=1)
    first_selections = []
    second_selections = []

    # act
    for _ in range(4):
        first_selections.append(selector_one.select_file())
        second_selections.append(selector_two.select_file())

    # assert
    assert first_selections == second_selections
