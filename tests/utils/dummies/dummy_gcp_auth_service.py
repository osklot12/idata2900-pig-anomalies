class DummyGCPAuthService:
    """A dummy GCPAuthService for testing, bypassing real authentication."""

    def get_access_token(self):
        """Returns a mock access token for testing."""
        return "dummy_access_token"