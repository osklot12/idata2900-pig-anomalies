import pytest

from src.data.dataset.matching.base_name_matching_strategy import BaseNameMatchingStrategy
from src.data.dataset.providers.simple_dataset_instance_provider import SimpleDatasetInstanceProvider
from src.data.dataset.selection.random_file_selector import RandomFileSelector
from tests.utils.dummies.dummy_dataset_source import DummyDatasetSource


@pytest.fixture
def dataset_source():
    """Fixture to provide a DummyDatasetSource instance."""
    return DummyDatasetSource()

@pytest.fixture
def video_suffixes():
    """Fixture to provide suffixes for video files."""
    return [
        "mp4",
        "mov"
    ]

@pytest.fixture
def annotation_suffixes():
    """Fixture to provide suffixes for annotation files."""
    return [
        "json",
        "txt"
    ]

@pytest.fixture
def video_selector(video_suffixes):
    """Fixture to provide a RandomFileSelectionStrategy."""
    return RandomFileSelector(video_suffixes)

@pytest.fixture
def annotation_matcher(annotation_suffixes):
    """Fixture to provide a BaseNameMatchingStrategy."""
    return BaseNameMatchingStrategy(annotation_suffixes)

def test_get_random_returns_file_pair(dataset_source, video_selector, annotation_matcher):
    """Tests that get_random() returns a file pair when a match exists."""
    # arrange
    files = [
        "dir_a/dir_b/file1.mp4",
        "dir_a/dir_b/file2.mp4",
        "dir_a/dir_b/file3.json",
        "dir_c/dir_e/file2.json"
    ]
    dataset_source.set_file_paths(files)

    file_pair_provider = SimpleDatasetInstanceProvider(dataset_source, video_selector, annotation_matcher)

    # act
    result = file_pair_provider.get_dataset_instance()

    # assert
    assert result is not None

def test_get_random_called_consecutively_gives_file_pairs(dataset_source, video_selector, annotation_matcher):
    """Tests that get_random() returns the expected amount of file pairs when called consecutively."""
    # arrange
    files = [
        "dir_a/dir_b/file1.mp4",
        "dir_a/dir_b/file2.mp4",
        "dir_c/dir_e/file1.json",
        "dir_c/dir_e/file2.json",
        "dir_a/dir_b/.secret"
    ]
    dataset_source.set_file_paths(files)

    file_pair_provider = SimpleDatasetInstanceProvider(dataset_source, video_selector, annotation_matcher)
    result = []

    # act
    for _ in range(10):
        result.append(file_pair_provider.get_dataset_instance())

    # assert
    print(f"{result}")
    assert len(result) == 10
    for pair in result:
        assert pair is not None

def test_get_random_returns_none_on_no_match(dataset_source, video_selector, annotation_matcher):
    """Tests that get_random() returns a None when no match exists."""
    # arrange
    files = [
        "dir_a/dir_b/file1.mp4",
        "dir_a/dir_b/file2.json"
    ]
    dataset_source.set_file_paths(files)

    file_pair_provider = SimpleDatasetInstanceProvider(dataset_source, video_selector, annotation_matcher)

    # act
    result = file_pair_provider.get_dataset_instance()

    # assert
    assert result is None