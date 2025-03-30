from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar("T")

class Broker(ABC, Generic[T]):
    """A publisher in the observer pattern."""

    @abstractmethod
    def notify(self, notification: T) -> None:
        """
        Sends a notification to the subscribers.

        Args:
            notification (T): the notification to send
        """
        raise NotImplementedError