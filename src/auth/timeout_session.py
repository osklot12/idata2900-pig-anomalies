import requests
from google.auth.transport.requests import Request

class TimeoutSession(requests.Session):
    """Session that times out."""

    def __init__(self, timeout: float = 30):
        """
        Initializes a TimeoutSession instance.

        Args:
            timeout (float): the timeout in seconds
        """
        super().__init__()
        self._timeout = timeout

    def request(self, *args, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", self._timeout)
        return super().request(*args, **kwargs)