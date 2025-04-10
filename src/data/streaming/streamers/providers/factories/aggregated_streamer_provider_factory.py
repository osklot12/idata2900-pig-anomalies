from src.data.streaming.streamers.providers.aggregated_streamer_provider import AggregatedStreamerProvider
from src.data.streaming.streamers.providers.factories.streamer_pair_provider_factory import StreamerPairProviderFactory
from src.data.streaming.streamers.providers.factories.streamer_provider_factory import StreamerProviderFactory
from src.data.streaming.streamers.providers.streamer_provider import StreamerProvider


class AggregatedStreamerProviderFactory(StreamerProviderFactory):
    """Factory for creating AggregatedStreamerProvider instances."""

    def __init__(self, pair_provider_factory: StreamerPairProviderFactory):
        """
        Initializes an AggregatedStreamerProviderFactory instance.

        Args:
            pair_provider_factory (StreamerPairProviderFactory): factory for creating streamer pair providers
        """
        self._pair_provider_factory = pair_provider_factory

    def create_provider(self) -> StreamerProvider:
        return AggregatedStreamerProvider(self._pair_provider_factory.create_pair_provider())