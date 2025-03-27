import socket
import pickle
import struct
import threading
from typing import List

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.network.ddp_sender_interface import DDPSenderInterface


class PipelineDataSender(DDPSenderInterface):
    """
    Sends augmented data (frames + annotations) from the pipeline
    to the worker machines for DDP training.
    """

    def __init__(self, worker_ips: List[str], port: int = 50051):
        self.worker_ips = worker_ips
        self.port = port
        self.sockets = {}

    def connect_to_workers(self):
        """Establish connections with worker PCs."""
        for ip in self.worker_ips:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((ip, self.port))
                self.sockets[ip] = s
            except Exception as e:
                print(f"[PipelineDataSender] Connection failed: {ip} ({e})")

    def send_data(self, data_batch: List[AnnotatedFrame]):
        if not self.sockets:
            print("[PipelineDataSender] No workers connected!")

        else:
            thread = threading.Thread(
                target=self._worker_thread,
                args=(data_batch,)
            )
            thread.start()

    def _worker_thread(self, data_batch: List[AnnotatedFrame]):
        for frame_annotation in data_batch:
            for worker_ip in self.worker_ips:
                if frame_annotation.annotations:
                    self._send_data(worker_ip, frame_annotation)

    def _send_data(self, worker_ip: str, data: AnnotatedFrame):
        """Send one frame + annotation to a worker."""
        try:
            packed_data = pickle.dumps(data)
            message = struct.pack(">L", len(packed_data)) + packed_data
            self.sockets[worker_ip].sendall(message)
            print(f"Sent {len(packed_data)} bytes to {worker_ip}")
        except Exception as e:
            print(f"[PipelineDataSender] Error sending data to {worker_ip}: {e}")

    def disconnect_workers(self):
        """Close all connections."""
        for ip, s in self.sockets.items():
            s.close()
        self.sockets.clear()
