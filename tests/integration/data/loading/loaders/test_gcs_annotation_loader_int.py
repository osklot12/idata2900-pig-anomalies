import pytest

from src.auth.gcp_auth_service import GCPAuthService
from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.decoders.darwin_decoder import DarwinDecoder
from src.data.loading.loaders.gcs_annotation_loader import GCSAnnotationLoader
from src.data.label.simple_label_parser import SimpleLabelParser
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from tests.utils.gcs.test_bucket import TestBucket


@pytest.fixture
def decoder():
    """Fixture to provide an AnnotationDecoder instance."""
    return DarwinDecoder(SimpleLabelParser(NorsvinBehaviorClass.get_label_map()))


@pytest.fixture
def gcs_annotation_loader(decoder):
    """Fixture to provide a GCSAnnotationLoader instance."""
    auth_service = GCPAuthService(TestBucket.SERVICE_ACCOUNT_FILE)
    return GCSAnnotationLoader(
        bucket_name=TestBucket.BUCKET_NAME,
        auth_service=auth_service,
        decoder=decoder
    )


@pytest.mark.integration
def test_load_annotation_success(gcs_annotation_loader):
    """Integration test for successfully loading an annotation from GCS."""
    # arrange
    annotation_id = TestBucket.SAMPLE_ANNOTATION

    # act
    annotations = gcs_annotation_loader.load_video_annotations(annotation_id)

    # assert
    assert isinstance(annotations, list)
    assert len(annotations) > 0
    assert all(isinstance(a, FrameAnnotations) for a in annotations)


@pytest.mark.integration
def test_load_annotation_not_found(gcs_annotation_loader):
    """Integration test for handling a missing annotation file in GCS."""
    # arrange
    annotation_id = "non_existent_annotation.json"

    # act & assert
    with pytest.raises(Exception, match="404 Client Error"):
        gcs_annotation_loader.load_video_annotations(annotation_id)
