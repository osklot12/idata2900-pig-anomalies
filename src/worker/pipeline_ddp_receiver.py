import socket
import pickle
import struct
import threading
import traceback
import multiprocessing
from src.worker.ddp_receiver_interface import DDPReceiverInterface
from src.worker.tensor_converter import convert_to_tensor_format
from src.worker.data_queue import data_queue


class PipelineDataReceiver(DDPReceiverInterface):
    """
    Receives image frames and annotations, then forwards them to the training system.
    """

    def __init__(self, host="0.0.0.0", port=50051):
        self.host = host
        self.port = port
        self.server_socket = None

    def start(self):
        """Start the receiver and listen for incoming data."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)

        while True:
            client_socket, _ = self.server_socket.accept()
            thread = threading.Thread(target=self.handle_client, args=(client_socket,))
            thread.start()

    def handle_client(self, client_socket):
        """Handles incoming frame & annotation data from the sender."""
        try:
            while True:
                data = self.receive_data(client_socket)

                if data is not None:
                    frame = data["frame"]  # np.ndarray (H, W, 3)
                    annotations = data["annotations"]  # List[Dict] like [{"bbox": [...], "label": ...}]

                    # Convert to PyTorch tensor format for Faster R-CNN
                    image_tensor, target = convert_to_tensor_format(frame, annotations)

                    # Push into the shared training queue
                    data_queue.put((image_tensor, target))


        except Exception as e:

            print("[Receiver] Exception in handle_client:")

            traceback.print_exc()

        finally:
            client_socket.close()

    def receive_data(self, client_socket):
        """Receives and unpacks data from the sender."""
        data_length = client_socket.recv(4)
        if not data_length:
            return None

        try:
            message_size = struct.unpack(">L", data_length)[0]
            data = b""

            while len(data) < message_size:
                packet = client_socket.recv(4096)
                if packet:
                    data += packet
                else:
                    return None

            return pickle.loads(data)
        except Exception:
            return None
