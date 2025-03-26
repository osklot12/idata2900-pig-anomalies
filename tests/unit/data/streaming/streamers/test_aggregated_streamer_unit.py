from unittest.mock import Mock

import pytest

from src.data.streaming.factories.streamer_pair_factory import StreamerPairFactory
from src.data.streaming.streamers.aggregated_streamer import AggregatedStreamer
from src.data.streaming.streamers.annotation_streamer import AnnotationStreamer
from src.data.streaming.streamers.video_streamer import VideoStreamer


@pytest.fixture
def dummy_callback():
    """Fixture to provide a dummy callback."""
    return Mock()


@pytest.fixture
def dummy_streamers():
    """Fixture to provide dummy streamers."""
    video_streamer = Mock(spec=VideoStreamer)
    annotation_streamer = Mock(spec=AnnotationStreamer)
    return video_streamer, annotation_streamer


@pytest.fixture
def streamer_pair_factory(dummy_streamers):
    """Fixture to provide a mock streamer pair factory."""
    factory = Mock(spec=StreamerPairFactory)
    factory.create_streamer_pair.return_value = dummy_streamers
    return factory


@pytest.fixture
def aggregated_streamer(streamer_pair_factory, dummy_callback):
    """Fixture to provide an AggregatedStreamer instance."""
    return AggregatedStreamer(streamer_pair_factory, dummy_callback)


@pytest.mark.unit
def test_start_streaming(aggregated_streamer, dummy_streamers):
    """Tests that start_streaming() streams successfully."""
    # arrange
    video_streamer, annotation_streamer = dummy_streamers

    # act
    aggregated_streamer.start_streaming()

    # assert
    video_streamer.start_streaming.assert_called_once()
    annotation_streamer.start_streaming.assert_called_once()

    aggregated_streamer.stop_streaming()


@pytest.mark.unit
def test_wait_for_completion(aggregated_streamer, dummy_streamers):
    """Tests that wait_for_completion() is called on the underlying streamers correctly."""
    # arrange
    video_streamer, annotation_streamer = dummy_streamers

    # act
    aggregated_streamer.start_streaming()
    aggregated_streamer.wait_for_completion()

    # assert
    video_streamer.wait_for_completion.assert_called_once()
    annotation_streamer.wait_for_completion.assert_called_once()

    aggregated_streamer.stop_streaming()


@pytest.mark.unit
def test_stop_streaming(aggregated_streamer, dummy_streamers):
    """Tests that stop_streaming() stops the underlying streamers correctly."""
    # arrange
    video_streamer, annotation_streamer = dummy_streamers

    # act
    aggregated_streamer.start_streaming()
    aggregated_streamer.stop_streaming()

    # assert
    video_streamer.stop_streaming.assert_called_once()
    annotation_streamer.stop_streaming.assert_called_once()


@pytest.mark.unit
def test_raises_when_streamer_pair_is_none(dummy_callback):
    """Tests that AggregatedStreamer raises RuntimeError if the factory returns None."""
    # arrange
    factory = Mock(spec=StreamerPairFactory)
    factory.create_streamer_pair.return_value = None

    # act & assert
    with pytest.raises(RuntimeError, match="Failed to create streamers"):
        AggregatedStreamer(factory, dummy_callback)
