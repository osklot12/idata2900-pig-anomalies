import threading
import time
from src.data.streamers.streamer import Streamer


class DummyStreamer(Streamer):
    """A dummy streamer for testing."""

    def __init__(self, wait_time: float = 0.1):
        self.wait_time = wait_time
        self.running = False
        self.lock = threading.Lock()

    def stream(self):
        with self.lock:
            self.running = True

    def stop(self):
        with self.lock:
            self.running = False

    def wait_for_completion(self):
        if self.running:
            time.sleep(self.wait_time)