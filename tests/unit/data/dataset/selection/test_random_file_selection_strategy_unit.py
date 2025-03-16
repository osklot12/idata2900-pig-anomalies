import pytest

from src.data.dataset.selection.random_file_selection_strategy import RandomFileSelectionStrategy

@pytest.fixture
def suffixes():
    """Returns a list of file suffixes."""
    return [
        "mp4",
        "mp3",
        "txt",
        "docx"
    ]

@pytest.fixture
def random_selector(suffixes):
    """Fixture to provide a RandomFileSelectionStrategy instance."""
    return RandomFileSelectionStrategy(suffixes)

def test_select_file_returns_file_with_valid_suffix(random_selector, suffixes):
    """Tests that select_file() returns a file with a valid suffix."""
    # arrange
    invalid_file_one = "file1.exe"
    invalid_file_two = "file2.dll"
    valid_file = "file3.mp3"
    files = [
        invalid_file_one,
        invalid_file_two,
        valid_file
    ]

    # act
    result = random_selector.select_file(files)

    # assert
    assert result is not None
    assert result == valid_file

def test_select_file_returns_none_on_no_valid_files(random_selector, suffixes):
    """Tests that select_file() returns None when no valid files are given."""
    # arrange
    files = [
        "file1.exe",
        "file2.dll"
    ]

    # act
    result = random_selector.select_file(files)

    # assert
    assert result is None

def test_select_file_is_random(random_selector, suffixes):
    """Tests that select_file() returns a file picked randomly."""
    # arrange
    files = [
        "file1.mp3",
        "file2.mp4",
        "file3.txt"
    ]
    result = []
    runs = 5
    iterations = 20

    # act
    for _ in range (runs):
        picked_files = []
        for _ in range(iterations):
            picked_files.append(random_selector.select_file(files))

        result.append(picked_files)

    # NOTE ABOUT NON-DETERMINISM:
    # with 5 iterations of picking 20 files out of 3 available files,
    # the chance of all lists being equal is ((1/3)^20)^5 = 1/(3^100)
    # which is practically 0. Thus, we can consider a failed test to indicate a true negative.

    # assert
    print(f"{result}")
    assert len(set(map(tuple, result))) > 1