import multiprocessing
import queue
import socket
import pickle
import struct
import threading
import traceback

from src.worker.data_provider import DataProvider
from src.worker.ddp_receiver_interface import DDPReceiverInterface
from src.worker.tensor_converter import convert_to_tensor_format
from src.data.dataclasses.annotated_frame import AnnotatedFrame


class PipelineDataReceiver(DDPReceiverInterface, DataProvider):
    """
    Receives image frames and annotations, then forwards them to the training system.
    """

    def __init__(self, host="0.0.0.0", port=50051):
        self.host = host
        self.port = port
        self.server_socket = None
        self._queue = multiprocessing.Manager().Queue()

    def handle_client(self, client_socket):
        """Handles incoming AnnotatedFrame objects from the sender."""
        frame_count = 0
        try:
            while True:
                annotated = self.receive_data(client_socket)

                if annotated is not None:
                    if not isinstance(annotated, AnnotatedFrame):
                        print("[Receiver] Warning: received object is not AnnotatedFrame")
                        continue

                    image_tensor, target = convert_to_tensor_format(annotated)
                    self._queue.put((image_tensor, target))
                    frame_count += 1
                    print(f"[Receiver] ✅ Frame {frame_count} received and queued.")
                else:
                    print("[Receiver] ❌ Failed to deserialize frame (possibly partial stream)")

        except Exception:
            print("[Receiver] Exception in handle_client:")
            traceback.print_exc()

        finally:
            client_socket.close()
        print("[Receiver] Closed connection.")

    def receive_data(self, client_socket):
        """Receives and unpacks a pickled AnnotatedFrame object from the sender."""
        try:
            # First receive 4 bytes (message length)
            data_length_bytes = b""
            while len(data_length_bytes) < 4:
                more = client_socket.recv(4 - len(data_length_bytes))
                if not more:
                    return None
                data_length_bytes += more

            message_size = struct.unpack(">L", data_length_bytes)[0]

            # Now receive the actual data
            data = b""
            while len(data) < message_size:
                packet = client_socket.recv(min(4096, message_size - len(data)))
                if not packet:
                    return None
                data += packet

            return pickle.loads(data)

        except Exception:
            traceback.print_exc()
            return None

    def get_queue(self) -> multiprocessing.Queue:
        return self._queue


    def start(self):
        """Start the receiver and listen for incoming data."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)

        print(f"[Receiver] Listening on {self.host}:{self.port}")

        while True:
            client_socket, client_addr = self.server_socket.accept()
            print(f"[Receiver] Connection from {client_addr}")
            thread = threading.Thread(target=self.handle_client, args=(client_socket,))
            thread.start()

