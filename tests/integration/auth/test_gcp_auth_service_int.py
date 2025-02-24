import pytest
import os
from src.auth.gcp_auth_service import GCPAuthService
from tests.conftest import get_test_path

TEST_CREDENTIALS_PATH = get_test_path(".secrets/service-account.json")

@pytest.fixture(scope="module")
def gcp_auth_service():
    """Creates a GCPAuthService instance using real credentials."""
    return GCPAuthService(credentials_path=TEST_CREDENTIALS_PATH)

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