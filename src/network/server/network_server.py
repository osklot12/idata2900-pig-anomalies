LISTENING_PORT = 50051

class NetworkServer:
    """A network server listening for incoming requests."""

    def __init__(self):
        """Initializes a NetworkServer instance."""
        self._running = False


    def run(self) -> None:
        """Runs the server."""
        if self._running:
            raise RuntimeError("Server already running")



    def _listen(self) ->  None:
        """Listens to incoming requests."""
        while


    def stop(self) -> None:
        """Stops the server."""