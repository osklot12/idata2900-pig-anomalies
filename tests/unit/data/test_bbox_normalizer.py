from unittest.mock import MagicMock

import pytest

from src.data.annotation_enum_parser import AnnotationEnumParser
from src.data.preprocessing.normalization.simple_bbox_normalizer import SimpleBBoxNormalizer


@pytest.fixture
def mock_annotation_parser():
    """Creates a mock annotation parser that converts string labels to enums."""
    mock_parser = MagicMock(spec=AnnotationEnumParser)

    mock_parser.enum_from_str.side_effect = lambda label: f"ENUM_{label.upper()}"

    return mock_parser

@pytest.fixture
def annotations():
    """Returns a list of annotations for testing purposes."""
    return [
        ("g2b_bellynosing", 576, 324, 192, 108),
        ("g2b_tailbiting", 1920, 1080, 288, 162)
    ]

@pytest.fixture
def expected_normalized_annotations(annotations):
    """Returns a list of expected normalized annotations for range [0, 1]"""
    return [
        ("ENUM_G2B_BELLYNOSING", 0.3, 0.3, 0.1, 0.1),
        ("ENUM_G2B_TAILBITING", 1.0, 1.0, 0.15, 0.15)
    ]

def test_normalize_annotations(mock_annotation_parser, annotations, expected_normalized_annotations):
    """Tests that annotations are correctly normalized and converted to enums."""
    # arrange
    image_dimensions = (1920, 1080)
    new_range = (0, 1)
    normalizer = SimpleBBoxNormalizer(image_dimensions, new_range, mock_annotation_parser)

    # act
    normalized_annotations = normalizer.normalize_annotations(annotations)

    # assert
    assert normalized_annotations == expected_normalized_annotations, \
        f"Expected: {expected_normalized_annotations}, got: {normalized_annotations}"

def test_normalize_annotations_swap_range(mock_annotation_parser, annotations, expected_normalized_annotations):
    """Tests that the new range values are swapped automatically when given in the wrong order."""
    # arrange
    image_dimensions = (1920, 1080)
    new_range = (1, 0)
    normalizer = SimpleBBoxNormalizer(image_dimensions, new_range, mock_annotation_parser)

    # act
    normalized_annotations = normalizer.normalize_annotations(annotations)

    # assert
    assert normalized_annotations == expected_normalized_annotations, \
        f"Expected: {expected_normalized_annotations}, got: {normalized_annotations}"

def test_normalize_annotations_without_parser(annotations):
    """Tests that an error is raised if annotation_parser is None."""
    # arrange
    image_dimensions = (1920, 1080)
    new_range = (0, 1)
    normalizer = SimpleBBoxNormalizer(image_dimensions, new_range, None)

    # act & assert
    with pytest.raises(ValueError, match="An annotation parser must be provided to map string labels to enums."):
        normalizer.normalize_annotations(annotations)