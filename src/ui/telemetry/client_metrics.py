import queue
import time

class ClientMetrics:
    """Metrics for a network client."""

    def __init__(self):
        """Initializes a ClientMetrics instance."""
        self._connected_at: float = time.time()
        self._requests: queue.Queue[float] = queue.Queue(maxsize=100)

    def get_connection_age(self) -> float:
        """
        Returns the age of the connection, in seconds.

        Returns:
            float: the age of the connection
        """
        return time.time() - self._connected_at