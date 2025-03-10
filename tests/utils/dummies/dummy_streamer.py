import threading
import time
from typing import List

from src.command.command import Command
from src.data.streamers.streamer import Streamer


class DummyStreamer(Streamer):
    """A dummy streamer for testing."""

    def __init__(self, wait_time: float = 0.1):
        self.wait_time = wait_time
        self.eos_commands: List[Command] = []
        self.stream_lock = threading.Lock()
        self.command_lock = threading.Lock()
        self._thread = None
        self._running = False

    def stream(self):
        with self.stream_lock:
            self._running = True
            self._thread = threading.Thread(target=self._test_worker)
            self._thread.start()

    def streaming(self) -> bool:
        with self.stream_lock:
            return self._running

    def _test_worker(self):
        time.sleep(self.wait_time)
        with self.stream_lock:
            self._running = False

        for cmd in self.eos_commands:
            cmd.execute()

    def stop(self):
        with self.stream_lock:
            self._thread.join()
            self._running = False

    def wait_for_completion(self):
        self.stop()

    def add_eos_command(self, command: Command) -> None:
        with self.command_lock:
            self.eos_commands.append(command)