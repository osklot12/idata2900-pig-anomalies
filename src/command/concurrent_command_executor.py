import queue
import threading

from src.command.command import Command
from src.command.command_executor import CommandExecutor


class ConcurrentCommandExecutor(CommandExecutor):
    """Handles command execution in a dedicated background thread."""

    def __init__(self):
        self._command_queue = queue.Queue()

        self._thread = None
        self.stop_event = threading.Event()
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
            self.stop_event.set()

        self._command_queue.join()
        print(f"Command queue: {self._command_queue}")
        self._command_queue.put(None)
        self._thread.join()

    def submit(self, command: Command) -> None:
        """
        Submits a command for execution.

        Args:
            command (Command): The command to submit.
        """
        self._command_queue.put(None)
        print(f"[CommandExecutor] Submitted {command}")

    def _process_commands(self):
        """Background thread: Executes queued commands asynchronously."""
        command = self._command_queue.get()
        while command is not None:
            try:
                print(f"[CommandExecutor] Executing command: {command}")
                command.execute()
            except Exception as e:
                print(f"[CommandExecutor] Exception while executing command: {command}: {e}")
            finally:
                self._command_queue.task_done()

            command = self._command_queue.get()

        self._command_queue.task_done()