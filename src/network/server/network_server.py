import threading

LISTENING_PORT = 50051

class NetworkServer:
    """A network server listening for incoming requests."""

    def __init__(self):
        """Initializes a NetworkServer instance."""
        self._running = False
        self._listen_thread = None
        self._lock = threading.Lock()


    def run(self) -> None:
        """Runs the server."""
        if self._is_running():
            raise RuntimeError("Server already running")


        self._listen_thread = threading.Thread(target=self._listen)
        self._listen_thread.start()

    def _listen(self) ->  None:
        """Listens to incoming requests."""
        while self._is_running():
            

    def stop(self) -> None:
        """Stops the server."""

    def _set_running(self, running: bool) -> None:
        with self._lock:
            self._running = running

    def _is_running(self) -> bool:
        with self._lock:
            return self._running