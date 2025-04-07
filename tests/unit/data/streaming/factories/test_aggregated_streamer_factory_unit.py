from unittest.mock import Mock

import pytest

from src.data.streaming.factories.aggregated_streamer_factory import AggregatedStreamerFactory
from src.data.streaming.streamers.aggregated_streamer import AggregatedStreamer


@pytest.fixture
def streamer_pair_factory():
    """Fixture to provide a mock StreamerPairFactory."""
    frame_streamer = Mock()
    annotation_streamer = Mock()

    factory = Mock()
    factory.create_streamer_pair.return_value = (frame_streamer, annotation_streamer)

    return factory


@pytest.fixture
def consumer():
    """Fixture to provide a mock Feedable instance."""
    return Mock()


@pytest.mark.unit
def test_aggregated_streamer_factory_returns_aggregated_streamer(streamer_pair_factory, consumer):
    """Tests that create_streamer() returns an AggregatedStreamer instance."""
    # arrange
    factory = AggregatedStreamerFactory(streamer_pair_factory)

    # act
    streamer = factory.create_streamer(consumer)

    # assert
    assert isinstance(streamer, AggregatedStreamer)
