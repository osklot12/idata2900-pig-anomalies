import pytest
from src.data.decoders.darwin_decoder import DarwinDecoder
from tests.utils.dummies.dummy_gcp_data_loader import DummyGCPDataLoader

@pytest.fixture
def data_loader():
    """Returns a DummyGCPDataLoader instance."""
    return DummyGCPDataLoader()

def _get_expected_annotations():
    """Returns the expected annotations for the DummyGCPDataLoader."""
    expected_annotations = {}

    for behavior, frames, in DummyGCPDataLoader.ANNOTATIONS:
        for frame, x, y, w, h in frames:
            frame_idx = int(frame)
            if frame_idx not in expected_annotations:
                expected_annotations[frame_idx] = []
            expected_annotations[frame_idx].append((behavior, x, y, w, h))

    return expected_annotations

def test_get_annotations(data_loader):
    """Tests that get_annotations returns the expected annotations."""
    # arrange
    sample_json = data_loader.download_json("test_annotations.json")
    decoder = DarwinDecoder(sample_json)

    # act
    decoded_annotations = decoder.get_annotations()

    # assert
    expected_annotations = _get_expected_annotations()

    assert len(expected_annotations) == len(decoded_annotations)

    for frame_idx, expected_list in expected_annotations.items():
        assert len(decoded_annotations[frame_idx]) == len(expected_list)
        for (expected_behavior, expected_x, expected_y, expected_w, expected_h), (
        actual_behavior, actual_x, actual_y, actual_w, actual_h) in zip(expected_list, decoded_annotations[frame_idx]):
            assert expected_behavior == actual_behavior
            assert expected_x == pytest.approx(actual_x, rel=1e-6)
            assert expected_y == pytest.approx(actual_y, rel=1e-6)
            assert expected_w == pytest.approx(actual_w, rel=1e-6)
            assert expected_h == pytest.approx(actual_h, rel=1e-6)

def test_get_frame_count(data_loader):
    """Tests that get_frame_count returns the expected frame count."""
    # arrange
    sample_json = data_loader.download_json("test_annotations.json")
    decoder = DarwinDecoder(sample_json)

    # act
    frame_count = decoder.get_frame_count()

    # assert
    assert frame_count == DummyGCPDataLoader.DEFAULT_FRAME_COUNT

def test_get_frame_dimensions(data_loader):
    """Tests that get_frame_dimensions returns the expected frame dimensions."""
    # arrange
    sample_json = data_loader.download_json("test_annotations.json")
    decoder = DarwinDecoder(sample_json)

    # act
    dim = decoder.get_frame_dimensions()

    # assert
    assert dim[0] == DummyGCPDataLoader.DEFAULT_FRAME_WIDTH
    assert dim[1] == DummyGCPDataLoader.DEFAULT_FRAME_HEIGHT