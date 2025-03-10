import pytest

from src.command.concurrent_command_executor import ConcurrentCommandExecutor
from tests.utils.dummies.dummy_command import DummyCommand
from tests.utils.dummies.dummy_failing_command import DummyFailingCommand


@pytest.fixture
def executor():
    """Fixture that provides a fresh CommandExecutor for each test."""
    return ConcurrentCommandExecutor()


def test_command_executor_starts_and_stops(executor):
    """Tests that the executor starts and stops correctly."""
    # act & assert
    executor.run()
    assert executor._thread is not None
    assert executor._thread.is_alive()

    executor.stop()
    assert executor._thread is None


def test_single_command_execution(executor):
    """Tests that a single command is executed."""
    # arrange
    command = DummyCommand()
    executor.run()

    # act
    executor.submit(command)

    # assert
    executor.stop()
    assert command.executed


def test_multiple_command_execution(executor):
    """Tests that multiple commands are executed in order."""
    # arrange
    commands = [DummyCommand() for _ in range(5)]
    executor.run()

    # act
    for cmd in commands:
        executor.submit(cmd)

    # assert
    executor.stop()
    assert all(cmd.executed for cmd in commands)

def test_executor_handles_failing_commands(executor):
    """Tests that the executor does not crash when a command fails."""
    # arrange
    failing_command = DummyFailingCommand()
    successful_command = DummyCommand()
    executor.run()

    # act
    executor.submit(failing_command)
    executor.submit(successful_command)

    # assert
    executor.stop()
    assert successful_command.executed