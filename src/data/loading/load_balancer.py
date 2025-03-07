from abc import abstractmethod

from src.data.streamers.streamer import Streamer


class LoadBalancer:
    """An interface for load balancers."""

    @abstractmethod
    def run(self):
        """Runs the load balancer."""
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        """Stops the load balancer."""
        raise NotImplementedError

    @abstractmethod
    def terminate_streamer(self, streamer: Streamer):
        """
        Terminates a streamer.

        Args:
            streamer: The streamer to terminate.
        """
        raise NotImplementedError