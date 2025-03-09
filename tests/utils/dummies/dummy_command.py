from src.command.command import Command


class DummyCommand(Command):
    """A dummy command that records if it was executed."""

    def __init__(self):
        self.executed = False

    def execute(self):
        self.executed = True