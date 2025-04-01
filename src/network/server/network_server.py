import threading
import socket

from src.network.messages.requests.handlers.registry.request_handler_registry import RequestHandlerRegistry
from src.network.messages.serialization.factories.deserializer_factory import DeserializerFactory
from src.network.messages.serialization.factories.serializer_factory import SerializerFactory
from src.network.network_config import NETWORK_SERVER_PORT
from src.network.server.client_handler import ClientHandler


class NetworkServer:
    """A network server listening for incoming requests."""

    def __init__(self, serializer_factory: SerializerFactory, deserializer_factory: DeserializerFactory,
                 handler_registry: RequestHandlerRegistry):
        """Initializes a NetworkServer instance."""
        self._serializer_factory = serializer_factory
        self._deserializer_factory = deserializer_factory
        self._handler_registry = handler_registry

        self._running = False
        self._listen_thread = None
        self._lock = threading.Lock()
        self._server_sock = None

    def run(self) -> None:
        """Runs the server."""
        with self._lock:
            if self._running:
                raise RuntimeError("Server already running")

            self._running = True

        self._listen_thread = threading.Thread(target=self._listen)
        self._listen_thread.start()

    def _listen(self) -> None:
        """Listens to incoming requests."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("0.0.0.0", NETWORK_SERVER_PORT))
            sock.listen()
            self._server_sock = sock

            print(f"[NetworkServer] Listening on port {NETWORK_SERVER_PORT}...")

            self._accept_clients()

    def _accept_clients(self) -> None:
        """Accepts incoming client connections."""
        while self._is_running():
            try:
                client_sock, addr = self._server_sock.accept()
                print(f"[NetworkServer] Connection from {addr}")
                self._handle_client(client_sock)
            except Exception as e:
                print(f"[NetworkServer] Error accepting client: {e}")

    def _handle_client(self, client_sock):
        """Creates and runs a handler for a client socket."""
        handler = ClientHandler(
            client_socket=client_sock,
            serializer=self._serializer_factory.create_serializer(),
            deserializer=self._deserializer_factory.create_deserializer(),
            handler_registry=self._handler_registry
        )
        threading.Thread(target=handler.handle, daemon=True).start()

    def stop(self) -> None:
        """Stops the server."""
        self._set_running(False)

        try:
            with socket.create_connection(("localhost", NETWORK_SERVER_PORT)) as sock:
                pass
        except Exception:
            pass

        if self._listen_thread:
            self._listen_thread.join()
        print("[NetworkServer] Server stopped.")

    def _set_running(self, running: bool) -> None:
        with self._lock:
            self._running = running

    def _is_running(self) -> bool:
        with self._lock:
            return self._running
