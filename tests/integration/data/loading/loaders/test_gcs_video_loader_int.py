import pytest

from src.auth.gcp_auth_service import GCPAuthService
from src.data.loading.loaders.gcs_video_loader import GCSVideoLoader
from tests.utils.gcs.test_bucket import TestBucket


@pytest.fixture
def gcs_video_loader():
    """Fixture to provide a GCSVideoLoader instance."""
    auth_service = GCPAuthService(TestBucket.SERVICE_ACCOUNT_FILE)
    return GCSVideoLoader(
        bucket_name=TestBucket.BUCKET_NAME,
        auth_service=auth_service
    )

@pytest.mark.integration
def test_load_video_success(gcs_video_loader):
    """Integration test for successfully loading a video from GCS."""
    # arrange
    video_id = TestBucket.SAMPLE_VIDEO

    # act
    video_content = gcs_video_loader.load_video_file(video_id)

    # assert
    assert isinstance(video_content, bytearray)
    assert len(video_content) > 0

@pytest.mark.integration
def test_load_video_not_found(gcs_video_loader):
    """Integration test for handling a missing video file in GCS."""
    # arrange
    video_id = "non_existent_video.mp4"

    # act & assert
    with pytest.raises(Exception, match="404 Client Error"):
        gcs_video_loader.load_video_file(video_id)