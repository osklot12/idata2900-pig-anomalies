import pytest

from src.auth.gcp_auth_service import GCPAuthService
from src.data.loading.loaders.gcs_annotation_loader import GCSAnnotationLoader
from tests.utils.gcs.test_bucket import TestBucket


@pytest.fixture
def gcs_annotation_loader():
    """Fixture to provide a GCSAnnotationLoader instance."""
    auth_service = GCPAuthService(TestBucket.SERVICE_ACCOUNT_FILE)
    return GCSAnnotationLoader(
        bucket_name=TestBucket.BUCKET_NAME,
        auth_service=auth_service
    )

@pytest.mark.integration
def test_load_annotation_success(gcs_annotation_loader):
    """Integration test for successfully loading an annotation from GCS."""
    # arrange
    annotation_id = TestBucket.SAMPLE_ANNOTATION

    # act
    annotation = gcs_annotation_loader.load_video_annotations(annotation_id)

    # assert
    assert isinstance(annotation, dict)
    assert len(annotation) > 0

@pytest.mark.integration
def test_load_annotation_not_found(gcs_annotation_loader):
    """Integration test for handling a missing annotation file in GCS."""
    # arrange
    annotation_id = "non_existent_annotation.json"

    # act & assert
    with pytest.raises(Exception, match="404 Client Error"):
        gcs_annotation_loader.load_video_annotations(annotation_id)