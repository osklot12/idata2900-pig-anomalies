import os

from google.auth.transport.requests import Request
from google.oauth2 import service_account

from src.auth.auth_service import AuthService


class GCPAuthService(AuthService):
    """Handles authentication and token management for Google Cloud Platform."""

    def __init__(self, credentials_path: str):
        """
        Initializes the authentication service.

        :param credentials_path: Path to the GCP service account JSON credentials file.
        """
        self.credentials_path = credentials_path
        self.token = None
        self.creds = None
        self.authenticate()

    def authenticate(self) -> None:
        """Authenticates with GCP using a service account."""
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(f"Service account file not found: {self.credentials_path}")

        self.creds = service_account.Credentials.from_service_account_file(
            self.credentials_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.refresh_token()

    def refresh_token(self) -> None:
        """Refreshes the token if needed."""
        if not self.creds or not self.creds.valid:
            self.creds.refresh(Request())
        self.token = self.creds.token

    def get_access_token(self) -> str:
        self.refresh_token()
        return self.token