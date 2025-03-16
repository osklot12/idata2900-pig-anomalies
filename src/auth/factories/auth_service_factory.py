from abc import ABC, abstractmethod

from src.auth.auth_service import AuthService


class AuthServiceFactory(ABC):
    """A factory interface for creating authentication services."""

    @abstractmethod
    def create_auth_service(self) -> AuthService:
        """
        Creates an AuthService instance.

        Returns:
            AuthService: the authentication service
        """
        raise NotImplementedError