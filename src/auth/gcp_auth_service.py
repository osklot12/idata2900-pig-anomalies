import os
import time

from google.auth.exceptions import TransportError
from google.auth.transport.requests import Request
from google.oauth2 import service_account

from src.auth.auth_service import AuthService
from src.auth.timeout_session import TimeoutSession


class GCPAuthService(AuthService):
    """Handles authentication and token management for Google Cloud Platform."""

    def __init__(self, credentials_path: str, max_retries: int = 5, timeout: int = 30):
        """
        Initializes the authentication service.

        :param credentials_path: Path to the GCP service account JSON credentials file.
        """
        self.credentials_path = credentials_path
        self.token = None
        self.creds = None
        self.max_retries = max_retries
        self.timeout = timeout
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

            session = TimeoutSession(timeout=self.timeout)
            request = Request(session=session)

            attempt = 1
            success = False

            while attempt <= self.max_retries and not success:
                try:
                    self.creds.refresh(request)
                    self.token = self.creds.token
                    success = True
                except TransportError as e:
                    if attempt < self.max_retries:
                        time.sleep(2 ** attempt)
                    attempt += 1

            if not success:
                raise RuntimeError(f"[GCPAuthService] Failed to refresh token after {self.max_retries} attempts.")

        self.token = self.creds.token

    def get_access_token(self) -> str:
        self.refresh_token()
        return self.token