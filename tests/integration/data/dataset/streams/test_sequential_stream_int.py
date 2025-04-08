import pytest

from src.auth.factories.gcp_auth_service_factory import GCPAuthServiceFactory
from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.dataset.factories.lazy_entity_factory import LazyEntityFactory
from src.data.dataset.manifests.matching_manifest import MatchingManifest
from src.data.dataset.providers.manifest_instance_provider import ManifestInstanceProvider
from src.data.dataset.registries.suffix_file_registry import SuffixFileRegistry
from src.data.dataset.selectors.determ_string_selector import DetermStringSelector
from src.data.dataset.splitters.determ_splitter import DetermSplitter
from src.data.dataset.streams.dock_stream import DockStream
from src.data.decoders.factories.darwin_decoder_factory import DarwinDecoderFactory
from src.data.label.factories.simple_label_parser_factory import SimpleLabelParserFactory
from src.data.loading.factories.gcs_loader_factory import GCSLoaderFactory
from src.data.parsing.base_name_parser import BaseNameParser
from src.data.preprocessing.normalization.factories.simple_bbox_normalizer_factory import SimpleBBoxNormalizerFactory
from src.data.preprocessing.resizing.factories.static_frame_resizer_factory import StaticFrameResizerFactory
from src.data.streaming.factories.aggregated_streamer_factory import AggregatedStreamerFactory
from src.data.streaming.factories.file_streamer_pair_factory import FileStreamerPairFactory
from src.data.streaming.managers.docking_streamer_manager import DockingStreamerManager
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
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
    return DetermSplitter(strings=manifest.ids, weights=[0.8, 0.1, 0.1])


@pytest.fixture
def instance_provider(manifest, splitter):
    """Fixture to provide an InstanceProvider instance."""
    return ManifestInstanceProvider(
        manifest=manifest,
        selector=DetermStringSelector(strings=splitter.splits[2])
    )

@pytest.fixture
def stream():
    """Fixture to provide a SequentialStream instance."""
    return DockStream()

@pytest.fixture
def entity_factory(loader_factory):
    """Fixture to provide a EntityFactory instance."""
    return LazyEntityFactory(loader_factory, BaseNameParser())

@pytest.fixture
def streamer_pair_factory(instance_provider, entity_factory):
    """Fixture to provide a StreamerPairFactory instance."""
    return FileStreamerPairFactory(
        instance_provider=instance_provider,
        entity_factory=entity_factory,
        frame_resizer_factory=StaticFrameResizerFactory((300, 300)),
        bbox_normalizer_factory=SimpleBBoxNormalizerFactory((0, 1))
    )

@pytest.fixture
def streamer_factory(streamer_pair_factory):
    """Fixture to provide an AggregatedStreamerFactory instance."""
    return AggregatedStreamerFactory(streamer_pair_factory)

@pytest.fixture
def manager(streamer_factory, stream):
    """Fixture to provide a SequentialStreamerManager instance."""
    return DockingStreamerManager(
        streamer_factory=streamer_factory,
        stream=stream
    )

def test_streaming_test_set(stream, manager):
    """Tests that the test set stream gives instances deterministically."""
    # arrange
    manager.run()

    instance = stream.read()
    while instance:
        assert isinstance(instance, StreamedAnnotatedFrame)
        instance = stream.read()
        StreamedAnnotatedFrameVisualizer.visualize(instance)
    print(f"Finished reading!")
    manager.stop()

