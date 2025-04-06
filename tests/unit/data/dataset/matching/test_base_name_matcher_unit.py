import pytest

from src.data.dataset.matching.base_name_matcher import BaseNameMatcher


@pytest.fixture
def matcher():
    """Fixture to provide a BaseNameMatcherStrategy instance."""
    return BaseNameMatcher()


@pytest.mark.unit
@pytest.mark.parametrize(
    "ref, candidates, expected_target",
    [
        (
                "dir_a/dir_b/instance1.mp4",
                [
                    "dir_a/dir_b/instance2.mp4", "dir_a/dir_b/instance3.json", "dir_c/dir_d/instance1.txt"
                ], "dir_c/dir_d/instance1.txt"
        ),
        (
                "a/b/c/video001.mp4",
                [
                    "a/b/c/video0001.mp4", "b/e/video001.mp4", "b/e/f/video1.json"
                ], "b/e/video001.mp4"
        ),
        (
                "abc",
                [
                    "abc/efg", "a/b/c/hij", "abc/abc"
                ], "abc/abc"
        )
    ]
)
def test_match_returns_matching_string_on_success(matcher, ref, candidates, expected_target):
    """Tests that match returns the matching string when such a match exists."""
    # act
    match = matcher.match(ref, candidates)

    # assert
    assert match == expected_target


@pytest.mark.unit
def test_match_returns_none_on_no_match(matcher):
    """Tests that match returns None when no match is found."""
    # arrange
    ref = "dir_a/dir_b/instance1.mp4"
    candidates = [
        "dir_a/dir_b/instance2.mp4", "dir_a/dir_b/instance3.json", "dir_c/dir_d/instance3.txt"
    ]

    # act
    match = matcher.match(ref, candidates)

    # assert
    assert match is None


@pytest.mark.unit
def test_find_match_raises_exception_on_none_file(matcher):
    """Tests that find_match() raises an exception when the file provided is None."""
    # arrange
    expected_error_msg = "file_name cannot be None"

    # act
    with pytest.raises(ValueError) as exc_info:
        matcher.match(None, ["some_file.json"])

    # assert
    assert str(exc_info.value) == expected_error_msg


@pytest.mark.unit
def test_find_match_raises_exception_on_none_candidates(matcher):
    """Tests that find_match() raises an exception when the candidates provided is None."""
    # arrange
    expected_error_msg = "candidates cannot be None"

    # act
    with pytest.raises(ValueError) as exc_info:
        matcher.match("some_file.mp4", None)

    # assert
    assert str(exc_info.value) == expected_error_msg
