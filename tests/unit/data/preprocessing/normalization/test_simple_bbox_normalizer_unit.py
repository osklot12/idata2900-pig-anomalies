import pytest

from src.data.dataclasses.bounding_box import BoundingBox
from src.data.preprocessing.normalization.normalizers.simple_bbox_normalizer import SimpleBBoxNormalizer

@pytest.fixture
def bboxes():
    """Returns a list of annotations for testing purposes."""
    return [
        BoundingBox(576, 324, 192, 108),
        BoundingBox(1920, 1080, 288, 162)
    ]

@pytest.fixture
def expected_normalized_bboxes(bboxes):
    """Returns a list of expected normalized annotations for range [0, 1]"""
    return [
        BoundingBox(0.3, 0.3, 0.1, 0.1),
        BoundingBox(1.0, 1.0, 0.15, 0.15)
    ]

def test_normalize_bboxes(bboxes, expected_normalized_bboxes):
    """Tests that normalize_bounding_box() normalizes values as expected."""
    # arrange
    normalizer = SimpleBBoxNormalizer(
        (1920, 1080), (0, 1)
    )
    normalized_bboxes = []

    # act
    for bbox in bboxes:
        normalized_bboxes.append(normalizer.normalize_bounding_box(bbox))

    # assert
    assert normalized_bboxes == expected_normalized_bboxes

def test_normalizer_swaps_range_for_decreasing_values(bboxes, expected_normalized_bboxes):
    """Tests that the range values gets automatically swapped when provided in decreasing order."""
    # arrange
    normalizer = SimpleBBoxNormalizer(
        (1920, 1080), (1, 0)
    )
    normalized_bboxes = []

    # act
    for bbox in bboxes:
        normalized_bboxes.append(normalizer.normalize_bounding_box(bbox))

    # assert
    assert normalized_bboxes == expected_normalized_bboxes