from typing import TypeVar, Generic, List

from src.observer.component.component_listener import ComponentListener
from src.observer.component.schema.schema import Schema

T = TypeVar('T', bound=Schema)


class ComponentEventBroker(Generic[T]):
    """Observer pattern broker for component information."""

    def __init__(self, component_id: str):
        """
        Initializes the ComponentInfoBroker.

        Args:
            component_id (str): the component id
        """
        self._id = component_id
        self._subscribers: List[ComponentListener[T]] = []

    def subscribe(self, subscriber: ComponentListener[T]) -> None:
        """
        Subscribes a listener to receive updates about component.

        Args:
            subscriber (ComponentListener[T]): the listener to subscribe
        """
        self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber: ComponentListener[T]) -> None:
        """
        Unsubscribes a listener to stop receiving updates.

        Args:
            subscriber (ComponentListener[T]): the listener to unsubscribe
        """
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)

    def notify(self, schema: T) -> None:
        """
        Notifies all subscribers about the new schema.

        Args:
            schema (T): the new schema
        """
        for subscriber in self._subscribers:
            subscriber.new_schema(self._id, schema)