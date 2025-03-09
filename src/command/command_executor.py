import queue
import threading

from src.command.command import Command


class CommandExecutor:
    """Handles command execution in a dedicated background thread."""

    def __init__(self):
        self.command_queue = queue.Queue()
        self.stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def run(self) -> None:
        """Runs the CommandExecutor worker thread."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise RuntimeError("Command executor already running.")

            self.stop_event.clear()
            self._thread = threading.Thread(target=self._process_commands)
            self._thread.start()

    def stop(self) -> None:
        """Stops the CommandExecutor worker thread."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                self.stop_event.set()
                self.command_queue.put(None)
                self._thread.join()
                self._thread = None

    def submit(self, command: Command) -> None:
        """
        Submits a command for execution.

        Args:
            command (Command): The command to submit.
        """
        self.command_queue.put(command)

    def _process_commands(self):
        """Background thread: Executes queued commands asynchronously."""
        command = self.command_queue.get()
        while command is not None:
            try:
                print(f"Executing command: {command}")
                command.execute()
            except Exception as e:
                print(f"[CommandExecutor] Exception while executing command: {command}: {e}")
            finally:
                self.command_queue.task_done()
            command = self.command_queue.get()