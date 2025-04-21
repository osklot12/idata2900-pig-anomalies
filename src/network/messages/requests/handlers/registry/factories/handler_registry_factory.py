from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.network.messages.requests.handlers.registry.request_handler_registry import RequestHandlerRegistry
from src.network.server.session.session import Session

T = TypeVar("T")


class HandlerRegistryFactory(Generic[T], ABC):
    """Interface for RequestHandlerRegistry factories."""

    @abstractmethod
    def create_registry(self, session: Session[T]) -> RequestHandlerRegistry:
        """
        Creates and returns a new RequestHandlerRegistry instance.

        Args:
            session (Session[T]): the `Session` instance to use for the handlers

        Returns:
            RequestHandlerRegistry: a new RequestHandlerRegistry instance
        """
        raise NotImplementedError