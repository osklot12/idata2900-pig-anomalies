from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.network.server.session.session import Session

T = TypeVar("T")


class SessionFactory(Generic[T], ABC):
    """Interface for session factories."""

    @abstractmethod
    def create_session(self, client_address: str) -> Session[T]:
        """
        Creates and returns a `Session` instance.

        Args:
            client_address (str): the address of the client

        Returns:
            Session[T]: the `Session` instance
        """
        raise NotImplementedError