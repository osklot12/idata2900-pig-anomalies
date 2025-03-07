from src.command.command import Command
from src.data.loading.load_balancer import LoadBalancer
from src.data.streamers.streamer import Streamer


class TerminateStreamerCommand(Command):
    """A command pattern object that terminates streamers."""

    def __init__(self, streamer: Streamer, balancer: LoadBalancer):
        """
        Initializes a new TerminateStreamerCommand object.

        Args:
            streamer (Streamer): The Streamer object to terminate.
            balancer (LoadBalancer): The LoadBalancer responsible for the streamer.
        """
        self.streamer = streamer
        self.balancer = balancer

    def execute(self):
        self.balancer.terminate_streamer(self.streamer)