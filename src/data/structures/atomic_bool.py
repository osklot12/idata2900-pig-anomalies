from src.data.structures.atomic_var import AtomicVar


class AtomicBool(AtomicVar[bool]):
    """Thread-safe boolean flag."""

    def __init__(self, initial: bool = False):
        """
        Initializes an AtomicBool instance.

        Args:
            initial (T): the initial value
        """
        super().__init__(initial)

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