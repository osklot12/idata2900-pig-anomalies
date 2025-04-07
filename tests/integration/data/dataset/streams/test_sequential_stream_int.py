import pytest

from src.auth.factories.gcp_auth_service_factory import GCPAuthServiceFactory
from src.data.dataset.manifests.matching_manifest import MatchingManifest
from src.data.dataset.providers.manifest_instance_provider import ManifestInstanceProvider
from src.data.dataset.registries.suffix_file_registry import SuffixFileRegistry
from src.data.dataset.selectors.determ_string_selector import DetermStringSelector
from src.data.dataset.splitters.determ_splitter import DetermSplitter
from src.data.dataset.streams.sequential_stream import SequentialStream
from src.data.decoders.factories.darwin_decoder_factory import DarwinDecoderFactory
from src.data.label.factories.simple_label_parser_factory import SimpleLabelParserFactory
from src.data.loading.factories.gcs_loader_factory import GCSLoaderFactory
from src.data.streaming.factories.aggregated_streamer_factory import AggregatedStreamerFactory
from src.data.streaming.factories.file_streamer_pair_factory import FileStreamerPairFactory
from src.data.streaming.factories.streamer_pair_factory import StreamerPairFactory
from src.data.streaming.managers.sequential_streamer_manager import SequentialStreamerManager
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from tests.utils.gcs.test_bucket import TestBucket


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
    return SequentialStream()

@pytest.fixture
def streamer_pair_factory():
    """Fixture to provide a StreamerPairFactory instance."""
    return FileStreamerPairFactory(

    )

@pytest.fixture
def streamer_factory():
    """Fixture to provide an AggregatedStreamerFactory instance."""
    return AggregatedStreamerFactory()

@pytest.fixture
def manager():
    """Fixture to provide a SequentialStreamerManager instance."""
    return SequentialStreamerManager()