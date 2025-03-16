from abc import ABC, abstractmethod

class AuthService(ABC):
    """Defines the interface for authentication services."""

    @abstractmethod
    def get_access_token(self) -> str:
        """
        Returns a valid access token.

        Returns:
            str: the access token
        """
        raise NotImplementedError