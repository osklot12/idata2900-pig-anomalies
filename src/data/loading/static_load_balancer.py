from typing import Type

from src.data.loading.load_balancer import LoadBalancer
from src.data.loading.streamer_provider import StreamerProvider


class StaticLoadBalancer(LoadBalancer):
    """A load balancer that manages a static number of workers."""

    def __init__(self, streamer_provider: Type[StreamerProvider], n_workers: int):
        """
        Initializes a new instance of the StaticLoadBalancer class.

        Args:
            streamer_provider (Type[StreamerProvider]): Provides streamers.
            n_workers (int): The number of workers to maintain. This is not the number of individual streamers, but number of logically grouped streamers.
        """
        self.streamer_provider = streamer_provider
        self.n_workers = n_workers

    def run(self):
        pass

    def stop(self):
        pass
