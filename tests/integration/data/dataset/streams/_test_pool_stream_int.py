import pytest

from src.auth.factories.gcp_auth_service_factory import GCPAuthServiceFactory
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset.providers.lazy_entity_factory import LazyEntityFactory
from src.data.dataset.manifests.matching_manifest import MatchingManifest
from src.data.dataset.providers.manifest_instance_provider import ManifestInstanceProvider
from src.data.dataset.registries.suffix_file_registry import SuffixFileRegistry
from src.data.dataset.selectors.random_string_selector import RandomStringSelector
from src.data.dataset.splitters.string_set_splitter import StringSetSplitter
from src.data.dataset.streams.factories.gcs_training_stream_factory import GCSTrainingStreamFactory
from src.data.dataset.streams.pool_stream import PoolStream
from src.data.decoders.factories.darwin_decoder_factory import DarwinDecoderFactory
from src.data.label.factories.simple_label_parser_factory import SimpleLabelParserFactory
from src.data.loading.factories.gcs_loader_factory import GCSLoaderFactory
from src.data.parsing.base_name_parser import BaseNameParser
from src.data.preprocessing.augmentation.augmentors.annotated_frame_augmentor import AnnotatedFrameAugmentor
from src.data.preprocessing.augmentation.augmentors.cond_augmentor import CondMultiplier
from src.data.preprocessing.normalization.factories.simple_bbox_normalizer_factory import SimpleBBoxNormalizerFactory
from src.data.preprocessing.resizing.factories.static_frame_resizer_factory import StaticFrameResizerFactory
from src.data.streaming.streamers.providers.aggregated_streamer_provider import AggregatedStreamerProvider
from src.data.streaming.streamers.providers.file_streamer_pair_provider import FileStreamerPairProvider
from src.data.streaming.managers.static_streamer_manager import StaticStreamerManager
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from src.utils.norsvin_dataset_config import NORSVIN_SPLIT_RATIOS
from tests.utils.gcs.test_bucket import TestBucket
from tests.utils.streamed_annotated_frame_visualizer import StreamedAnnotatedFrameVisualizer


@pytest.fixture
def auth_factory():
    """Fixture to provide an AuthServiceFactory instance."""
    return GCPAuthServiceFactory(TestBucket.SERVICE_ACCOUNT_FILE)


@pytest.fixture
def decoder_factory():
    """Fixture to provide a DecoderFactory instance."""
    return DarwinDecoderFactory(SimpleLabelParserFactory(NorsvinBehaviorClass.get_label_map()))


@pytest.fixture
def loader_factory(auth_factory, decoder_factory):
    """Fixture to provide a LoaderFactory instance."""
    return GCSLoaderFactory(
        bucket_name=TestBucket.BUCKET_NAME,
        auth_factory=auth_factory,
        decoder_factory=decoder_factory
    )


@pytest.fixture
def manifest(loader_factory):
    """Fixture to provide a Manifest instance."""
    source = loader_factory.create_file_registry()
    return MatchingManifest(
        video_registry=SuffixFileRegistry(source=source, suffixes=("mp4",)),
        annotations_registry=SuffixFileRegistry(source=source, suffixes=("json",))
    )


@pytest.fixture
def splitter(manifest):
    """Fixture to provide a DetermSplitter instance."""
    return StringSetSplitter(strings=manifest.ids, weights=[0.8, 0.1, 0.1])


@pytest.fixture
def instance_provider(manifest, splitter):
    """Fixture to provide an InstanceProvider instance."""
    return ManifestInstanceProvider(
        manifest=manifest,
        selector=RandomStringSelector(strings=splitter.splits[0])
    )


@pytest.fixture
def entity_factory(loader_factory):
    """Fixture to provide an EntityFactory instance."""
    return LazyEntityFactory(loader_factory, BaseNameParser())


@pytest.fixture
def streamer_pair_factory(instance_provider, entity_factory):
    """Fixture to provide a StreamerPairFactory instance."""
    return FileStreamerPairProvider(
        instance_provider=instance_provider,
        entity_factory=entity_factory,
        frame_resizer_factory=StaticFrameResizerFactory((1920, 1080)),
        bbox_normalizer_factory=SimpleBBoxNormalizerFactory((0, 1))
    )


@pytest.fixture
def streamer_factory(streamer_pair_factory):
    """Fixture to provide an AggregatedStreamerFactory instance."""
    return AggregatedStreamerProvider(streamer_pair_factory)


@pytest.fixture
def multiplier():
    """Fixture to provide a CondMultiplier instance."""

    def is_annotated(instance: AnnotatedFrame):
        return len(instance.annotations) > 0

    return CondMultiplier(50, is_annotated)


@pytest.fixture
def augmentor():
    """Fixture to provide an Augmentor instance."""
    return AnnotatedFrameAugmentor()


@pytest.fixture
def stream(multiplier, augmentor):
    """Fixture to provide a PoolStream instance."""
    return PoolStream[AnnotatedFrame](
        pool_size=1000,
        preprocessors=[multiplier, augmentor]
    )


@pytest.fixture
def manager(streamer_factory, stream):
    """Fixture to provide a StaticStreamerManager instance."""
    return StaticStreamerManager[AnnotatedFrame](
        streamer_factory=streamer_factory,
        consumer=stream
    )


def test_streaming_train_set(stream, manager):
    """Tests that the train set stream gives random training instances."""
    # arrange
    manager.run()

    # act
    instance = stream.read()
    i = 0
    while instance and i < 10000:
        assert isinstance(instance, AnnotatedFrame)
        instance = stream.read()
        StreamedAnnotatedFrameVisualizer.visualize(instance)
        i += 1
    print(f"Finished reading!")
    manager.stop()

def test_norsvin_train_stream():
    """Tests Norsvin training set stream."""
    # arrange
    factory = GCSTrainingStreamFactory(
        bucket_name=TestBucket.BUCKET_NAME,
        service_account_path=TestBucket.SERVICE_ACCOUNT_FILE,
        frame_size=(1920, 1080),
        label_map=NorsvinBehaviorClass.get_label_map(),
        split_ratios=NORSVIN_SPLIT_RATIOS,
        pool_size=1000
    )
    managed_stream = factory.create_stream()

    # act
    managed_stream.start()
    stream = managed_stream.stream
    instance = stream.read()
    i = 0
    while instance and i < 10000:
        assert isinstance(instance, AnnotatedFrame)
        instance = stream.read()
        StreamedAnnotatedFrameVisualizer.visualize(instance)
        i += 1
    print(f"Finished reading!")
    
    managed_stream.stop()