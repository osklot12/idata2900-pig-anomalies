import socket
import pickle
import struct
import threading
import numpy as np
from typing import List, Dict
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

    def send_data(self, frames: List[np.ndarray], annotations: List[List[Dict]]):
        """
        Sends a batch of frames & annotations to worker PCs.

        :param frames: List of image frames (numpy arrays).
        :param annotations: List of annotations corresponding to each frame.
        """
        if not self.sockets:
            print("[PipelineDataSender] No workers connected!")

        elif len(frames) != len(annotations):
            print("[PipelineDataSender] Frame-annotation mismatch!")

        else:
            for i in range(len(frames)):
                frame, ann = frames[i], annotations[i]
                for worker_ip in self.worker_ips:
                    thread = threading.Thread(
                        target=self._send_data,
                        args=(worker_ip, {"frame": frame, "annotations": ann}),
                    )
                    thread.start()

    def _send_data(self, worker_ip: str, data: Dict):
        """Send one frame + annotation to a worker."""
        try:
            packed_data = pickle.dumps(data)
            message = struct.pack(">L", len(packed_data)) + packed_data
            self.sockets[worker_ip].sendall(message)
        except Exception as e:
            print(f"[PipelineDataSender] Error sending data to {worker_ip}: {e}")

    def disconnect_workers(self):
        """Close all connections."""
        for ip, s in self.sockets.items():
            s.close()
        self.sockets.clear()
