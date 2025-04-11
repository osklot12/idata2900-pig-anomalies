import pytest

from src.auth.factories.gcp_auth_service_factory import GCPAuthServiceFactory
from src.data.dataset.providers.lazy_entity_factory import LazyEntityFactory
from src.data.dataset.matching.base_name_matcher import BaseNameMatcher
from src.data.dataset.selectors.random_string_selector import RandomStringSelector
from src.data.dataset.splitters.consistent_dataset_splitter import ConsistentDatasetSplitter
from src.data.dataset.virtual.frame_dataset import FrameDataset
from src.data.dataset.dataset_split import DatasetSplit
from src.data.decoders.factories.darwin_decoder_factory import DarwinDecoderFactory
from src.data.label.factories.simple_label_parser_factory import SimpleLabelParserFactory
from src.data.loading.factories.gcs_loader_factory import GCSLoaderFactory
from src.data.parsing.factories.base_name_parser_factory import BaseNameParserFactory
from src.data.parsing.base_name_parser import BaseNameParser
from src.data.preprocessing.normalization.factories.simple_bbox_normalizer_factory import SimpleBBoxNormalizerFactory
from src.data.preprocessing.resizing.factories.static_frame_resizer_factory import StaticFrameResizerFactory
from src.data.streaming.streamers.providers.file_streamer_pair_provider import FileStreamerPairProvider
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from tests.utils.gcs.test_bucket import TestBucket


@pytest.fixture
def auth_factory():
    """Fixture to provide a AuthServiceFactory instance."""
    return GCPAuthServiceFactory(TestBucket.SERVICE_ACCOUNT_FILE)


@pytest.fixture
def label_parser_factory():
    """Fixture to provide a LabelParserFactory instance."""
    return SimpleLabelParserFactory(NorsvinBehaviorClass.get_label_map())


@pytest.fixture
def decoder_factory(label_parser_factory):
    """Fixture to provide a DecoderFactory instance."""
    return DarwinDecoderFactory(label_parser_factory)


@pytest.fixture
def loader_factory(auth_factory, decoder_factory):
    """Fixture to provide a LoaderFactory instance."""
    return GCSLoaderFactory(
        bucket_name=TestBucket.BUCKET_NAME,
        auth_factory=auth_factory,
        decoder_factory=decoder_factory
    )


@pytest.fixture
def instance_provider(loader_factory):
    """Fixture to provide a DatasetInstanceProvider instance."""
    return SimpleDatasetInstanceProvider(
        source=loader_factory.create_file_registry(),
        video_selector=RandomStringSelector(["mp4"]),
        annotation_matcher=BaseNameMatcher(["json"])
    )


@pytest.fixture
def entity_factory(loader_factory):
    """Fixture to provide a DatasetEntityFactory instance."""
    return LazyEntityFactory(loader_factory, BaseNameParser())


@pytest.fixture
def frame_resizer_factory():
    """Fixture to provide a FrameResizerFactory instance."""
    return StaticFrameResizerFactory((2688, 1520))


@pytest.fixture
def bbox_normalizer_factory():
    """Fixture to provide a BBoxNormalizerFactory instance."""
    return SimpleBBoxNormalizerFactory((0, 1))


@pytest.fixture
def streamer_pair_factory(instance_provider, entity_factory, frame_resizer_factory, bbox_normalizer_factory):
    """Fixture to provide a StreamerPairFactory instance."""
    return FileStreamerPairProvider(
        instance_provider=instance_provider,
        entity_factory=entity_factory,
        frame_resizer_factory=frame_resizer_factory,
        bbox_normalizer_factory=bbox_normalizer_factory
    )


@pytest.fixture
def dataset_splitter():
    """Fixture to provide a DatasetSplitter instance."""
    return ConsistentDatasetSplitter()


@pytest.fixture
def virtual_dataset(dataset_splitter):
    """Fixture to provide a VirtualDataset instance."""
    return FrameDataset(
        splitter=dataset_splitter,
        max_sources=10
    )


@pytest.fixture
def source_parser_factory():
    """Fixture to provide a StringParserFactory instance."""
    return BaseNameParserFactory()


@pytest.fixture
def aggregated_streamer_factory(streamer_pair_factory, virtual_dataset, source_parser_factory):
    """Fixture to provide an AggregatedStreamerFactory instance."""
    return OldAggregatedStreamerFactory(
        streamer_pair_factory=streamer_pair_factory,
        callback=virtual_dataset.feed,
        source_parser_factory=source_parser_factory
    )


@pytest.mark.integration
def test_streaming_data(virtual_dataset, aggregated_streamer_factory):
    """Tests that data is successfully streamed from the could and stored in the VirtualDataset."""
    # arrange
    streamer = aggregated_streamer_factory.create_streamer()

    # act
    streamer.start_streaming()
    streamer.wait_for_completion()

    # assert
    n_frames_train = virtual_dataset.get_frame_count(DatasetSplit.TRAIN)
    n_frames_val = virtual_dataset.get_frame_count(DatasetSplit.VAL)
    n_frames_test = virtual_dataset.get_frame_count(DatasetSplit.TEST)

    assert n_frames_train or n_frames_val or n_frames_test
