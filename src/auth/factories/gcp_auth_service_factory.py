from src.auth.auth_service import AuthService
from src.auth.factories.auth_service_factory import AuthServiceFactory
from src.auth.gcp_auth_service import GCPAuthService


class GCPAuthServiceFactory(AuthServiceFactory):
    """A factory for creating Google Cloud Platform auth services."""

    def __init__(self, service_account_file: str):
        """
        Initializes a GCPAuthServiceFactory instance.

        Args:
            service_account_file (str): the path to the service account json file
        """
        self._service_account_file = service_account_file

    def create_auth_service(self) -> AuthService:
        return GCPAuthService(self._service_account_file)