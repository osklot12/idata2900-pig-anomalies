import pytest

from src.data.dataset.dataset_file_matcher import DatasetFileMatcher
from tests.utils.dummies.dummy_dataset_source import DummyDatasetSource


@pytest.fixture
def dataset_source():
    """Fixture for creating a dummy DatasetSource instance."""
    return DummyDatasetSource()


@pytest.fixture
def file_matcher(dataset_source):
    """Fixture for creating a DatasetEntryProvider instance."""
    video_suffixes = ["mp4", ".mov"]
    annotation_suffixes = ["json", "txt"]

    return DatasetFileMatcher(
        source=dataset_source,
        video_suffixes=video_suffixes,
        annotation_suffixes=annotation_suffixes
    )


def test_initialization(dataset_source):
    """Tests that initialization does not raise an exception."""
    matcher = DatasetFileMatcher(dataset_source, ["mp4"], ["json"])

    assert matcher is not None


def test_get_random_matches_one_pair(file_matcher):
    """Tests that get_random() successfully matches a pair and returns it."""
    # arrange
    n_unmatched_before = len(file_matcher._unmatched_files)

    # act
    pair = file_matcher.get_random()
    n_unmatched_after = len(file_matcher._unmatched_files)

    # assert
    assert pair is not None
    assert n_unmatched_before - n_unmatched_after == 2


def test_get_random_when_already_matched(file_matcher):
    """Tests that get_random() returns an already matched successfully."""
    # arrange
    # match all unmatched files first
    while file_matcher._unmatched_files:
        file_matcher.get_random()

    # act
    pair = file_matcher.get_random()

    # assert
    assert pair is not None
    print(f"{pair}")


def test_get_random_non_determinism(file_matcher):
    """Tests that get_random() acts in a non-deterministic manner."""
    # arrange
    runs = 5
    pairs_per_run = 20
    results = []

    # NOTE ABOUT NON-DETERMINISM:
    # with 5 iterations of picking 20 pairs out of 3 available pairs,
    # the chance of all lists being equal is ((1/3)^20)^5 = 1/(3^100)
    # which is practically 0. Thus we can consider a failed test to show non-determinism!

    # act
    for _ in range(runs):
        pairs = []
        for _ in range(pairs_per_run):
            pairs.append(file_matcher.get_random())
        results.append(pairs)

    # assert
    print(f"{results}")
    assert len(set(map(tuple, results))) > 1


def test_get_random_removes_unknown_files():
    """Tests that get_random() removes unknown files with the SilentRemovalStrategy."""
    # arrange
    provider = DummyDatasetSource()
    provider.set_file_paths(["dir_a/dir_b/id1.weird_extension"])
    matcher = DatasetFileMatcher(provider, ["mp4"], ["json"])

    # act
    matcher.get_random()

    # assert
    assert len(matcher._unmatched_files) == 0


def test_get_random_returns_none_when_no_files_available():
    """Tests that get_random() returns None if there are no matches."""
    # arrange
    provider = DummyDatasetSource()
    provider.set_file_paths([])
    matcher = DatasetFileMatcher(provider, ["mp4"], ["json"])

    # act
    pair = matcher.get_random()

    # assert
    assert pair is None


def test_get_random_removes_files_without_matches():
    """Tests that get_random() removes files without matches with the SilentRemovalStrategy."""
    # arrange
    provider = DummyDatasetSource()
    provider.set_file_paths([
        "dir_a/dir_b/id1.mp4",
        "dir_a/dir_c/id2.json"
    ])
    matcher = DatasetFileMatcher(provider, ["mp4"], ["json"])

    # act
    for _ in range(2):
        matcher.get_random()

    n_file_paths = matcher.n_file_paths()

    # assert
    assert n_file_paths == 0


def test_get_random_ignores_files_missing_suffix():
    """Tests that get_random() does not match files without suffix if not set explicitly."""
    # arrange
    provider = DummyDatasetSource()
    provider.set_file_paths([
        "dir_a/dir_b/id1.mp4",
        "dir_a/dir_b/id1"
    ])
    matcher = DatasetFileMatcher(provider, ["mp4"], ["json"])

    # act
    pair = matcher.get_random()

    # assert
    assert pair is None