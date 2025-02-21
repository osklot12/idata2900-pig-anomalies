import pytest
from unittest.mock import patch, MagicMock
from src.auth.gcp_auth_service import GCPAuthService

TEST_CREDENTIALS_PATH = "fake_servie_account.json"


@pytest.fixture
def mock_auth_service():
    """Creates a GCPAuthService instance with mocked authentication."""
    with patch("os.path.exists", return_value=True):
        with patch("src.auth.gcp_auth_service.service_account.Credentials.from_service_account_file") as mock_creds:
            mock_creds_instance = MagicMock()
            mock_creds_instance.token = "mock_access_token"
            mock_creds_instance.valid = True
            mock_creds_instance.refresh = MagicMock()

            mock_creds.return_value = mock_creds_instance

            auth_service = GCPAuthService(credentials_path=TEST_CREDENTIALS_PATH)
            auth_service.creds = mock_creds_instance
            return auth_service


def test_authentication(mock_auth_service):
    """Tests that authentication initializes correctly and a token is set."""
    assert mock_auth_service.creds is not None
    assert mock_auth_service.creds.token == "mock_access_token"

@patch("src.auth.gcp_auth_service.service_account.Credentials.from_service_account_file")
def test_refresh_token(mock_creds, mock_auth_service):
    """Tests if refresh_token correctly updates the token."""
    mock_creds_instance = MagicMock()
    mock_creds_instance.valid = False
    mock_creds_instance.token = "new_mock_token"
    mock_creds_instance.refresh = MagicMock(side_effect=lambda req: setattr(mock_creds_instance, "valid", True))

    mock_creds.return_value = mock_creds_instance
    mock_auth_service.creds = mock_creds.return_value

    mock_auth_service.refresh_token()
    assert mock_auth_service.creds.valid
    assert mock_auth_service.creds.token == "new_mock_token"

def test_get_access_token(mock_auth_service):
    """Tests if get_access_token correctly retrieves a valid token."""
    token = mock_auth_service.get_access_token()
    assert token == "mock_access_token"

@patch("os.path.exists", return_value=False)
def test_authenticate_with_missing_credentials(mock_exists):
    """Tests if authenticate raises FileNotFoundError when credentials are missing."""
    with pytest.raises(FileNotFoundError, match="Service account file not found"):
        GCPAuthService(credentials_path="nonexistent_file.json")