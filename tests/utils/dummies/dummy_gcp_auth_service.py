from src.auth.auth_service import AuthService


class DummyGCPAuthService(AuthService):
    """A dummy GCPAuthService for testing, bypassing real authentication."""

    def get_access_token(self) -> str:
        """Returns a mock access token for testing."""
        return "dummy_access_token"