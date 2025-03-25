import pytest

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.bbox import BBox
from src.data.decoders.darwin_decoder import DarwinDecoder
from src.data.label.simple_label_parser import SimpleLabelParser
from src.typevars.enum_type import T_Enum
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
            _create_anno_bbox(NorsvinBehaviorClass.BELLY_NOSING, 110, 100, 400, 300),
            _create_anno_bbox(NorsvinBehaviorClass.TAIL_BITING, 300, 250, 500, 600)
        ],
        60: [
            _create_anno_bbox(NorsvinBehaviorClass.BELLY_NOSING, 115, 105, 405, 305),
            _create_anno_bbox(NorsvinBehaviorClass.TAIL_BITING, 305, 255, 505, 605)
        ],
        140: [
            _create_anno_bbox(NorsvinBehaviorClass.BELLY_NOSING, 112, 126, 420, 320)
        ],
        144: [
            _create_anno_bbox(NorsvinBehaviorClass.BELLY_NOSING, 120, 130, 430, 330)
        ]
    }
    return {i: annotations.get(i, []) for i in range(expected_frame_count)}


def _create_anno_bbox(cls: T_Enum, center_x: float, center_y: float, width: float, height: float) -> AnnotatedBBox:
    """Creates a AnnotatedBBox instance."""
    return AnnotatedBBox(
        cls=cls,
        bbox=BBox(
            center_x=center_x,
            center_y=center_y,
            width=width,
            height=height
        )
    )


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


@pytest.mark.unit
def test_get_annotations(decoder, expected_sample_annotations, sample_json_bytes):
    """Tests that get_annotations returns the expected annotations."""
    # act
    decoded_list = decoder.decode_annotations(sample_json_bytes)

    # assert
    assert decoded_list
    assert len(decoded_list) == 208

    for decoded_frame_anno in decoded_list:
        assert decoded_frame_anno.source.source_id == "test-source-id"
        assert decoded_frame_anno.source.frame_resolution == (2688, 1520)

        decoded_annotations = decoded_frame_anno.annotations
        expected_annotations = expected_sample_annotations.get(decoded_frame_anno.index, [])

        assert len(decoded_annotations) == len(expected_annotations)

        for anno_bbox in decoded_annotations:
            assert anno_bbox in expected_annotations


@pytest.mark.unit
def test_get_frame_count(decoder, sample_json_bytes, expected_frame_count):
    """Tests that get_frame_count returns the expected frame count."""
    # act
    frame_count = decoder.get_frame_count(sample_json_bytes)

    # assert
    assert frame_count == expected_frame_count


@pytest.mark.unit
def test_get_frame_dimensions(decoder, sample_json_bytes):
    """Tests that get_frame_dimensions returns the expected frame dimensions."""
    # act
    frame_dimensions = decoder.get_frame_dimensions(sample_json_bytes)

    # assert
    assert frame_dimensions == (2688, 1520)
