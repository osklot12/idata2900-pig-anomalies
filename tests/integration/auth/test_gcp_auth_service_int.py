import pytest
from src.auth.gcp_auth_service import GCPAuthService
from src.utils.path_finder import PathFinder
from tests.utils.gcs.test_bucket import TestBucket

@pytest.fixture(scope="module")
def gcp_auth_service():
    """Creates a GCPAuthService instance using real credentials."""
    return GCPAuthService(
        str(PathFinder.get_abs_path(TestBucket.SERVICE_ACCOUNT_FILE))
    )

@pytest.mark.integration
def test_authenticate(gcp_auth_service):
    """Integration test: Ensures authentication initializes correctly with real credentials."""
    assert gcp_auth_service.creds is not None

@pytest.mark.integration
def test_get_access_token(gcp_auth_service):
    """Integration test: Retrieves a valid access token from GCP."""
    token = gcp_auth_service.get_access_token()
    assert isinstance(token, str)
    assert len(token) > 0

@pytest.mark.integration
def test_refresh_token(gcp_auth_service):
    """Integration test: Refreshes the token using a real API request."""
    gcp_auth_service.refresh_token()
    assert gcp_auth_service.creds.valid