from unittest.mock import Mock

import pytest

from src.auth.factories.gcp_auth_service_factory import GCPAuthServiceFactory
from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.dataset.providers.lazy_entity_provider import LazyEntityProvider
from src.data.dataset.matching.base_name_matcher import BaseNameMatcher
from src.data.dataset.selectors.random_string_selector import RandomStringSelector
from src.data.decoders.factories.darwin_decoder_factory import DarwinDecoderFactory
from src.data.label.factories.simple_label_parser_factory import SimpleLabelParserFactory
from src.data.loading.factories.gcs_loader_factory import GCSLoaderFactory
from src.data.parsing.base_name_parser import BaseNameParser
from src.data.preprocessing.normalization.factories.simple_bbox_normalizer_factory import SimpleBBoxNormalizerFactory
from src.data.preprocessing.resizing.factories.static_frame_resizer_factory import StaticFrameResizerFactory
from src.data.streaming.streamers.providers.file_streamer_pair_provider import FileStreamerPairProvider
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from tests.utils.gcs.test_bucket import TestBucket


@pytest.fixture
def auth_factory():
    """Fixture to provide an AuthServiceFactory."""
    return GCPAuthServiceFactory(TestBucket.SERVICE_ACCOUNT_FILE)


@pytest.fixture
def label_parser_factory():
    """Fixture to provide a LabelParserFactory instance."""
    return SimpleLabelParserFactory(NorsvinBehaviorClass.get_label_map())


@pytest.fixture
def decoder_factory(label_parser_factory):
    """Fixture to provide an AnnotationDecoderFactory instance."""
    return DarwinDecoderFactory(label_parser_factory)


@pytest.fixture
def loader_factory(auth_factory, decoder_factory):
    """Fixture to provide a LoaderFactory."""
    return GCSLoaderFactory(TestBucket.BUCKET_NAME, auth_factory, decoder_factory)


@pytest.fixture
def video_selector():
    """Fixture to provide a FileSelectionStrategy."""
    return RandomStringSelector(["mp4"])


@pytest.fixture
def annotation_matcher():
    """Fixture to provide a MatchingStrategy."""
    return BaseNameMatcher(["json"])


@pytest.fixture
def instance_provider(loader_factory, video_selector, annotation_matcher):
    """Fixture to provide a DatasetInstanceProvider instance."""
    return SimpleDatasetInstanceProvider(
        source=loader_factory.create_file_registry(),
        video_selector=video_selector,
        annotation_matcher=annotation_matcher,
    )


@pytest.fixture
def dataset_entity_factory(loader_factory):
    """Fixture to provide a DatasetEntityFactory instance."""
    return LazyEntityProvider(loader_factory, BaseNameParser())


@pytest.fixture
def resizer_factory():
    """Fixture to provide a FrameResizerFactory instance."""
    return StaticFrameResizerFactory((640, 640))


@pytest.fixture
def bbox_normalizer_factory():
    """Fixture to provide a BBoxNormalizerFactory instance."""
    return SimpleBBoxNormalizerFactory((0, 1))


@pytest.fixture
def streamer_pair_factory(instance_provider, dataset_entity_factory, resizer_factory, bbox_normalizer_factory):
    """Fixture to provide a StreamerPairFactory instance."""
    return FileStreamerPairProvider(
        instance_provider=instance_provider,
        entity_factory=dataset_entity_factory,
        frame_resizer_factory=resizer_factory,
        bbox_normalizer_factory=bbox_normalizer_factory
    )


@pytest.fixture
def callback():
    """Fixture to provide a mock callback."""
    return Mock()


def _validate_callbacks(callback: Mock):
    """Validates callbacks."""
    for call_args in callback.call_args_list:
        instance = call_args[0][0]
        assert isinstance(instance, StreamedAnnotatedFrame)
        assert instance.frame.size > 0


@pytest.mark.integration
def test_start_streaming(streamer_pair_factory, callback):
    """Tests that start_streaming() successfully streams video frames and annotations."""
    # arrange
    streamer = AggregatedStreamer(streamer_pair_factory, callback)

    # act
    streamer.start_streaming()
    streamer.wait_for_completion()

    # assert
    _validate_callbacks(callback)


@pytest.mark.integration
def test_start_streaming_parallel(streamer_pair_factory, callback):
    """Tests that start_streaming() for two AggregatedStreamers in parallel does not cause problems."""
    # arrange
    streamer = AggregatedStreamer(streamer_pair_factory, callback)

    another_callback = Mock()
    another_streamer = AggregatedStreamer(streamer_pair_factory, another_callback)

    # act
    streamer.start_streaming()
    another_streamer.start_streaming()

    streamer.wait_for_completion()
    another_streamer.wait_for_completion()

    # assert
    _validate_callbacks(callback)
    _validate_callbacks(another_callback)