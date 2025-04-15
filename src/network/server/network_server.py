import threading
import socket

from src.data.structures.atomic_bool import AtomicBool
from src.network.messages.requests.handlers.registry.factories.handler_registry_factory import HandlerRegistryFactory
from src.network.messages.serialization.factories.deserializer_factory import DeserializerFactory
from src.network.messages.serialization.factories.serializer_factory import SerializerFactory
from src.network.network_config import NETWORK_SERVER_PORT
from src.network.server.client_handler import ClientHandler
from src.network.server.session.factories.session_factory import SessionFactory


class NetworkServer:
    """A network server listening for incoming requests."""

    def __init__(self, serializer_factory: SerializerFactory, deserializer_factory: DeserializerFactory,
                 session_factory: SessionFactory, handler_factory: HandlerRegistryFactory):
        """
        Initializes a NetworkServer instance.

        Attributes:
            serializer_factory (SerializerFactory): factory for message serializers
            deserializer_factory (DeserializerFactory): factory for message deserializers
            session_factory (SessionFactory): factory for network sessions
            handler_factory (HandlerRegistryFactory): factory for request handler registries
        """
        self._serializer_factory = serializer_factory
        self._deserializer_factory = deserializer_factory
        self._session_factory = session_factory
        self._handler_factory = handler_factory

        self._running = AtomicBool(False)
        self._listen_thread = None
        self._lock = threading.Lock()
        self._server_sock = None

    def run(self) -> None:
        """Runs the server."""
        if self._running:
            raise RuntimeError("Server already running")

        self._running.set(True)

        self._listen_thread = threading.Thread(target=self._listen)
        self._listen_thread.start()

    def _listen(self) -> None:
        """Listens to incoming requests."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", NETWORK_SERVER_PORT))
            sock.listen()
            sock.settimeout(1.0)
            self._server_sock = sock

            print(f"[NetworkServer] Listening on port {NETWORK_SERVER_PORT}...")
            self._accept_clients()

    def _accept_clients(self) -> None:
        """Accepts incoming client connections."""
        while self._running:
            try:
                client_sock, addr = self._server_sock.accept()
                print(f"[NetworkServer] Connection from {addr}")
                self._handle_client(client_sock)

            except socket.timeout:
                continue

            except OSError as e:
                if self._running:
                    print(f"[NetworkServer] Socket error: {e}")

            except Exception as e:
                print(f"[NetworkServer] Error accepting client: {e}")

    def _handle_client(self, client_sock: socket.socket) -> None:
        """Creates and runs a handler for a client socket."""
        ip_address = client_sock.getpeername()[0]
        session = self._session_factory.create_session(client_address=ip_address)
        handler = ClientHandler(
            client_socket=client_sock,
            serializer=self._serializer_factory.create_serializer(),
            deserializer=self._deserializer_factory.create_deserializer(),
            handler_registry=self._handler_factory.create_registry(session),
            session=session
        )
        threading.Thread(target=handler.handle, daemon=True).start()

    def stop(self) -> None:
        """Stops the server."""
        print(f"[NetworkServer] Stopping server, please wait...")
        self._running.set(False)

        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception as e:
                print(f"[NetworkServer] Error closing server socket: {e}")

        if self._listen_thread:
            self._listen_thread.join()
        print("[NetworkServer] Server stopped.")