import queue
import threading
import time

from src.command.command import Command
from src.data.streamers.streamer import Streamer


class DummyStreamer(Streamer):
    """A dummy streamer for testing."""

    def __init__(self, wait_time: float = 0.1):
        self.wait_time = wait_time
        self.eos_commands = queue.Queue()
        self.lock = threading.Lock()
        self._thread = None

    def stream(self):
        with self.lock:
            if self._thread and self._thread.is_alive():
                raise RuntimeError("Streamer already running")
            self._thread = threading.Thread(target=self._test_worker)
            self._thread.start()

    def streaming(self) -> bool:
        with self.lock:
            return self._thread and self._thread.is_alive()

    def _test_worker(self):
        time.sleep(self.wait_time)
        while not self.eos_commands.empty():
            cmd = self.eos_commands.get()
            cmd.execute()

    def stop(self):
        with self.lock:
            if self._thread and self._thread.is_alive():
                self._thread.join()
            self._thread = None

    def wait_for_completion(self):
        self.stop()

    def add_eos_command(self, command: Command) -> None:
        self.eos_commands.put(command)