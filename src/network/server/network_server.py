import threading
import socket

from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.factories.gcs_eval_stream_factory import GCSEvalStreamFactory
from src.data.dataset.streams.factories.gcs_training_stream_factory import GCSTrainingStreamFactory
from src.network.messages.requests.get_batch_request import GetBatchRequest
from src.network.messages.requests.handlers.get_batch_handler import GetBatchHandler
from src.network.messages.requests.handlers.open_stream_handler import OpenStreamHandler
from src.network.messages.requests.handlers.registry.request_handler_registry import RequestHandlerRegistry
from src.network.messages.requests.open_stream_request import OpenStreamRequest
from src.network.messages.serialization.factories.deserializer_factory import DeserializerFactory
from src.network.messages.serialization.factories.serializer_factory import SerializerFactory
from src.network.network_config import NETWORK_SERVER_PORT
from src.network.server.client_handler import ClientHandler
from src.network.server.client_session import ClientSession
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from src.utils.norsvin_dataset_config import NORSVIN_SPLIT_RATIOS
from tests.utils.gcs.test_bucket import TestBucket


class NetworkServer:
    """A network server listening for incoming requests."""

    def __init__(self, serializer_factory: SerializerFactory, deserializer_factory: DeserializerFactory):
        """Initializes a NetworkServer instance."""
        self._serializer_factory = serializer_factory
        self._deserializer_factory = deserializer_factory

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
            handler_registry=self._create_request_handler_registry()
        )
        threading.Thread(target=handler.handle, daemon=True).start()

    @staticmethod
    def _create_request_handler_registry() -> RequestHandlerRegistry:
        """Creates a RequestHandlerRegistry instance for a new client."""
        session = ClientSession(
            streams={
                DatasetSplit.TRAIN: None,
                DatasetSplit.VAL: None,
                DatasetSplit.TEST: None
            }
        )
        registry = RequestHandlerRegistry()

        stream_factories = {
            DatasetSplit.TRAIN: GCSTrainingStreamFactory(
                bucket_name=TestBucket.BUCKET_NAME,
                service_account_path=TestBucket.SERVICE_ACCOUNT_FILE,
                frame_size=(640, 640),
                label_map=NorsvinBehaviorClass.get_label_map(),
                split_ratios=NORSVIN_SPLIT_RATIOS,
                pool_size=1000
            ),
            DatasetSplit.VAL: GCSEvalStreamFactory(
                bucket_name=TestBucket.BUCKET_NAME,
                service_account_path=TestBucket.SERVICE_ACCOUNT_FILE,
                split=1,
                frame_size=(640, 640),
                label_map=NorsvinBehaviorClass.get_label_map(),
                split_ratios=NORSVIN_SPLIT_RATIOS,
                buffer_size=3
            ),
            DatasetSplit.TEST: GCSEvalStreamFactory(
                bucket_name=TestBucket.BUCKET_NAME,
                service_account_path=TestBucket.SERVICE_ACCOUNT_FILE,
                split=2,
                frame_size=(640, 640),
                label_map=NorsvinBehaviorClass.get_label_map(),
                split_ratios=NORSVIN_SPLIT_RATIOS,
                buffer_size=3
            )
        }

        registry.register(OpenStreamRequest, OpenStreamHandler(session=session, stream_factories=stream_factories))
        registry.register(GetBatchRequest, GetBatchHandler(session=session))

        return registry

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
