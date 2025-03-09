from src.command.command import Command


class DummyFailingCommand(Command):
    """A dummy command that raises an exception when executed."""

    def execute(self):
        raise RuntimeError("Intentional failure")