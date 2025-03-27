import pytest

from src.auth.gcp_auth_service import GCPAuthService
from src.data.dataset.sources.gcs_source_registry import GCSSourceRegistry
from tests.utils.gcs.test_bucket import TestBucket


@pytest.fixture
def gcp_auth_service():
    """Fixture to provide a GCPAuthService instance."""
    return GCPAuthService(TestBucket.SERVICE_ACCOUNT_FILE)

@pytest.fixture
def gcs_file_manager(gcp_auth_service):
    """Fixture to provide a GCSFileManager instance."""
    return GCSSourceRegistry(
        bucket_name=TestBucket.BUCKET_NAME,
        auth_service=gcp_auth_service
    )

@pytest.mark.integration
def test_list_files_success(gcs_file_manager):
    """Integration test for successfully listing all files."""
    # act
    result = gcs_file_manager.get_source_ids()

    # assert
    assert isinstance(result, set)
    assert len(result) > 0

def test_list_files_bucket_not_found(gcp_auth_service):
    """Integration test for handling non-existent bucket."""
    # act
    file_manager = GCSSourceRegistry(
        bucket_name="non-existent-bucket-98542354082764032",
        auth_service=gcp_auth_service
    )

    # act & assert
    with pytest.raises(Exception, match="404 Client Error"):
        file_manager.get_source_ids()