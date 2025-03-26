import pytest

from src.data.dataset.matching.base_name_matching_strategy import BaseNameMatchingStrategy


@pytest.fixture
def suffixes():
    """Fixture to provide a list of valid suffixes for the matching file."""
    return [
        "json",
        "txt"
    ]


@pytest.fixture
def matcher(suffixes):
    """Fixture to provide a BaseNameMatcherStrategy instance."""
    return BaseNameMatchingStrategy(suffixes)


@pytest.mark.unit
def test_find_match_returns_matching_file(matcher):
    """Tests that find_match() returns a matching file successfully."""
    # arrange
    file = "dir_a/dir_b/file1.mp4"

    invalid_candidate_one = "dir_a/dir_b/file1.mp4"
    invalid_candidate_two = "dir_c/dir_e/file2.json"
    valid_candidate = "dir_c/dir_e/file1.json"
    candidates = [
        invalid_candidate_one,
        valid_candidate,
        invalid_candidate_two,
    ]

    # act
    result = matcher.find_match(file, candidates)

    # assert
    assert result == valid_candidate


@pytest.mark.unit
def test_find_match_returns_none_on_no_valid_match(matcher):
    """Tests that find_match() returns None when no valid match is found."""
    # arrange
    file = "dir_a/dir_b/single.mp4"

    candidates = [
        "dir_a/dir_b/single.mp4",
        "dir_a/dir_b/mingle.mp4"
        "dir_c/dir_e/secret_file.txt"
    ]

    # act
    result = matcher.find_match(file, candidates)

    # assert
    assert result is None


@pytest.mark.unit
def test_find_match_raises_exception_on_none_file(matcher):
    """Tests that find_match() raises an exception when the file provided is None."""
    # arrange
    expected_error_msg = "file_name cannot be None"

    # act
    with pytest.raises(ValueError) as exc_info:
        matcher.find_match(None, ["some_file.json"])

    # assert
    assert str(exc_info.value) == expected_error_msg


@pytest.mark.unit
def test_find_match_raises_exception_on_none_candidates(matcher):
    """Tests that find_match() raises an exception when the candidates provided is None."""
    # arrange
    expected_error_msg = "candidates cannot be None"

    # act
    with pytest.raises(ValueError) as exc_info:
        matcher.find_match("some_file.mp4", None)

    # assert
    assert str(exc_info.value) == expected_error_msg
