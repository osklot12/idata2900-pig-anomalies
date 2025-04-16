from unittest.mock import patch, MagicMock

import pytest
from requests.exceptions import HTTPError

from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.decoders.darwin_decoder import DarwinDecoder
from src.data.loading.loaders.gcs_annotation_loader import GCSAnnotationLoader
from src.data.dataset.label import SimpleLabelParser
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from src.utils.path_finder import PathFinder
from tests.utils.dummies.dummy_auth_service import DummyAuthService


@pytest.fixture
def decoder():
    """Fixture to provide an AnnotationDecoder instance."""
    return DarwinDecoder(SimpleLabelParser(NorsvinBehaviorClass.get_label_map()))


@pytest.fixture
def gcs_annotation_loader(decoder):
    """Fixture to provide a GCSAnnotationLoader instance."""
    return GCSAnnotationLoader("test-bucket", DummyAuthService(), decoder)


@pytest.fixture
def sample_json_bytes():
    """Fixture to provide a sample Darwin JSON annotation file."""
    test_file_path = PathFinder.get_abs_path("tests/data/annotations/test-darwin.json")
    with open(test_file_path, "r", encoding="utf-8") as f:
        return f.read().encode("utf-8")


@pytest.mark.unit
@patch.object(GCSAnnotationLoader, "_make_request")
def test_load_annotation_success(mock_make_request, gcs_annotation_loader, sample_json_bytes):
    """Tests successful loading of an annotation file."""
    # arrange
    annotation_id = "sample_annotation.json"

    mock_response = MagicMock()
    mock_response.content = sample_json_bytes
    mock_make_request.return_value = mock_response

    # act
    result = gcs_annotation_loader.load_video_annotations(annotation_id)
    print(f"{result}")

    # assert
    assert isinstance(result, list)
    assert len(result) > 0

    for annotations in result:
        assert isinstance(annotations, FrameAnnotations)


@pytest.mark.unit
@patch.object(GCSAnnotationLoader, "_make_request")
def test_load_annotation_not_found(mock_make_request, gcs_annotation_loader):
    """Tests handling when the requested annotation file is not found."""
    # arrange
    annotation_id = "missing_annotation.json"
    error_msg = "404 Client Error: Not Found"

    mock_make_request.side_effect = HTTPError(error_msg)

    # act & assert
    with pytest.raises(HTTPError, match=error_msg):
        gcs_annotation_loader.load_video_annotations(annotation_id)

    mock_make_request.assert_called_once_with(gcs_annotation_loader._get_file_url(annotation_id))
