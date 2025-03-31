import threading

class AtomicBool:
    """Thread-safe boolean flag."""

    def __init__(self, initial: bool = False):
        """
        Initializes an AtomicBool instance.

        Args:
            initial (bool): the initial value of the bool
        """
        self._value = initial
        self._lock = threading.Lock()

    def get(self) -> bool:
        """
        Returns the current value.

        Returns:
            bool: the current value
        """
        with self._lock:
            return self._value

    def set(self, value: bool) -> None:
        """
        Sets the current value.

        Args:
             value (bool): the new value
        """
        with self._lock:
            self._value = value

    def toggle(self) -> bool:
        """
        Toggles the current value.

        Returns:
            bool: the value after being toggled
        """
        with self._lock:
            self._value = not self._value
            return self._value

    def __bool__(self) -> bool:
        return self.get()