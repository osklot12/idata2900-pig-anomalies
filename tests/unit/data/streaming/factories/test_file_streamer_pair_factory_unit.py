from unittest.mock import Mock

import pytest

from src.data.dataclasses.dataset_instance import DatasetInstance
from src.data.streaming.factories.file_streamer_pair_factory import FileStreamerPairFactory
from src.data.streaming.streamers.video_annotations_streamer import VideoAnnotationsStreamer
from src.data.streaming.streamers.video_file_streamer import VideoFileStreamer


@pytest.fixture
def dataset_instance():
    """Fixture to provide a DatasetInstance instance."""
    return DatasetInstance(video_file="video.mp4", annotation_file="annotations.json")


@pytest.fixture
def instance_provider(dataset_instance):
    """Fixture to provide a mock DatasetInstanceProvider instance."""
    provider = Mock()
    provider.next.return_value = dataset_instance
    return provider


@pytest.fixture
def entity_factory():
    """Fixture to provide a mock DatasetEntityFactory instance."""
    factory = Mock()
    factory.create_video_file.return_value = Mock()
    factory.create_video_annotations.return_value = Mock()
    return factory


@pytest.fixture
def frame_resizer_factory():
    """Fixture to provide a mock FrameResizerFactory instance."""
    factory = Mock()
    factory.create_frame_resizer.return_value = Mock()
    return factory


@pytest.fixture
def bbox_normalizer_factory():
    """Fixture to provide a mock BBoxNormalizerFactory instance."""
    factory = Mock()
    factory.create_bbox_normalizer.return_value = Mock()
    return factory


@pytest.fixture
def streamer_pair_factory(instance_provider, entity_factory, frame_resizer_factory, bbox_normalizer_factory):
    """Fixture to provide a mock FileStreamerPairFactory instance."""
    return FileStreamerPairFactory(
        instance_provider=instance_provider,
        entity_factory=entity_factory,
        frame_resizer_factory=frame_resizer_factory,
        bbox_normalizer_factory=bbox_normalizer_factory
    )


@pytest.mark.unit
def test_create_streamer_pair_success(streamer_pair_factory):
    """Tests that calling create_streamer_pair() successfully creates a pair of video and annotation streamers."""
    # arrange
    frame_consumer = Mock()
    annotations_consumer = Mock()

    # act
    streamers = streamer_pair_factory.create_streamer_pair(frame_consumer, annotations_consumer)

    # assert
    assert streamers is not None
    video_streamer, annotation_streamer = streamers
    assert isinstance(video_streamer, VideoFileStreamer)
    assert isinstance(annotation_streamer, VideoAnnotationsStreamer)


@pytest.mark.unit
def test_create_streamer_pair_no_instance(entity_factory, frame_resizer_factory, bbox_normalizer_factory):
    """Tests that calling create_streamer_pair() returns None when no available instance is provided."""
    # arrange
    empty_provider = Mock()
    empty_provider.next.return_value = None
    factory = FileStreamerPairFactory(
        instance_provider=empty_provider,
        entity_factory=entity_factory,
        frame_resizer_factory=frame_resizer_factory,
        bbox_normalizer_factory=bbox_normalizer_factory
    )

    # act
    streamers = factory.create_streamer_pair(Mock(), Mock())

    # assert
    assert streamers is None
