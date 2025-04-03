import threading
from typing import TypeVar, Generic, Callable

T = TypeVar("T")

class AtomicVar(Generic[T]):
    """An atomic variable."""

    def __init__(self, initial: T):
        """
        Initializes an AtomicVar instance.

        Args:
            initial (T): the initial value
        """
        self._value = initial
        self._lock = threading.Lock()

    def get(self) -> T:
        """
        Returns the current value.

        Returns:
            T: the current value
        """
        with self._lock:
            return self._value

    def set(self, value: T) -> None:
        """
        Sets the current value.

        Args:
            value (T): the value to set
        """
        with self._lock:
            self._value = value

    def update(self, fn: Callable[[T], T]) -> None:
        """
        Atomically updates the value using the provided function.

        Args:
            fn (Callable[[T], T]): a function that takes the current value and returns the new value
        """
        with self._lock:
            self._value = fn(self._value)

    def __repr__(self) -> str:
        return f"AtomicVar({repr(self._value)})"