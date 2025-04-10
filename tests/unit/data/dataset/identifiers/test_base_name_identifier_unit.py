import pytest

from src.data.dataset.identifiers.base_name_identifier import BaseNameIdentifier


@pytest.fixture
def identifier():
    """Fixture to provide a BaseNameIdentifier instance."""
    return BaseNameIdentifier()


@pytest.mark.unit
@pytest.mark.parametrize(
    "video_path, annotations_path, expected_id",
    [
        ("dir_a/dir_b/instance1.mp4", "dir_b/dir_c/instance1.json", "instance1"),
        ("videos/foo/bar/abc.mp4", "annotations/xyz/abc.json", "abc"),
        ("a/b/c/video001.mp4", "x/y/z/video001.json", "video001")
    ]
)
def test_identify_success(identifier, video_path, annotations_path, expected_id):
    """Tests that identify successfully extracts a base name ID when the provided strings have matching base names."""
    # act
    id_ = identifier.identify(video_path, annotations_path)

    # assert
    assert id_ == expected_id


@pytest.mark.unit
def test_identify_base_name_mismatch_raises(identifier):
    """Tests that a basename mismatch between the video and annotations path causes a raise."""
    # arrange
    video_path = "dir_a/dir_b/instance1.mp4"
    annotations_path = "dir_a/dir_b/instance2.json"

    # act & assert
    with pytest.raises(ValueError):
        identifier.identify(video_path, annotations_path)
