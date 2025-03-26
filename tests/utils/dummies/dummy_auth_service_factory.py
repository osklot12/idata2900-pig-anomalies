from src.auth.auth_service import AuthService
from src.auth.factories.auth_service_factory import AuthServiceFactory
from tests.utils.dummies.dummy_auth_service import DummyAuthService


class DummyAuthServiceFactory(AuthServiceFactory):
    """A dummy authentication service factory for testing."""

    def create_auth_service(self) -> AuthService:
        return DummyAuthService()

