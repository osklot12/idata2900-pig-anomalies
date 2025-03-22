import pytest

from src.data.decoders.darwin_decoder import DarwinDecoder
from src.data.label.simple_label_parser import SimpleLabelParser
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from src.utils.path_finder import PathFinder


@pytest.fixture
def expected_frame_count():
    """Fixture to provide the expected frame count."""
    return 208


@pytest.fixture
def expected_sample_annotations(expected_frame_count):
    """Fixture to provide sampled expected annotations."""
    annotations = {
        56: [
            (110, 100, 400, 300, NorsvinBehaviorClass.BELLY_NOSING),
            (300, 250, 500, 600, NorsvinBehaviorClass.TAIL_BITING)
        ],
        60: [
            (115, 105, 405, 305, NorsvinBehaviorClass.BELLY_NOSING),
            (305, 255, 505, 605, NorsvinBehaviorClass.TAIL_BITING)
        ],
        140: [
            (112, 126, 420, 320, NorsvinBehaviorClass.BELLY_NOSING)
        ],
        144: [
            (120, 130, 430, 330, NorsvinBehaviorClass.BELLY_NOSING)
        ]
    }
    return {i: annotations.get(i, []) for i in range(expected_frame_count)}


@pytest.fixture
def sample_json_bytes():
    """Fixture to provide a sample Darwin JSON annotation file."""
    test_file_path = PathFinder.get_abs_path("tests/data/annotations/test-darwin.json")
    with open(test_file_path, "r", encoding="utf-8") as f:
        return f.read().encode("utf-8")


@pytest.fixture
def label_parser():
    """Returns a LabelParser instance."""
    return SimpleLabelParser(NorsvinBehaviorClass.get_label_map())


@pytest.fixture
def decoder(sample_json_bytes, label_parser):
    """Fixture to initialize DarwinDecoder with test data."""
    return DarwinDecoder(label_parser)


def test_get_annotations(decoder, expected_sample_annotations, sample_json_bytes):
    """Tests that get_annotations returns the expected annotations."""
    # act
    decoded_annotations = decoder.decode_annotations(sample_json_bytes)

    # assert
    assert decoded_annotations, "Decoded annotations should not be empty"

    decoded_dict = {}

    for frame in decoded_annotations:
        frame_index = frame.index
        decoded_dict[frame_index] = [
            (
                ann.bbox.center_x,
                ann.bbox.center_y,
                ann.bbox.width,
                ann.bbox.height,
                ann.cls
            )
            for ann in frame.annotations
        ]

    assert set(decoded_dict.keys()) == set(expected_sample_annotations.keys()), (
        f"Frame keys mismatch: Expected {set(expected_sample_annotations.keys())}, Got {set(decoded_dict.keys())}"
    )

    for frame_index, expected_annotations in expected_sample_annotations.items():
        assert decoded_dict[frame_index] == expected_annotations, (
            f"Mismatch at frame {frame_index}: Expected {expected_annotations}, Got {decoded_dict[frame_index]}"
        )

    print(f"{decoded_annotations}")


def test_get_frame_count(decoder, sample_json_bytes, expected_frame_count):
    """Tests that get_frame_count returns the expected frame count."""
    # act
    frame_count = decoder.get_frame_count(sample_json_bytes)

    # assert
    assert frame_count == expected_frame_count


def test_get_frame_dimensions(decoder, sample_json_bytes):
    """Tests that get_frame_dimensions returns the expected frame dimensions."""
    # act
    frame_dimensions = decoder.get_frame_dimensions(sample_json_bytes)

    # assert
    assert frame_dimensions == (2688, 1520)
