import pytest

from src.auth.factories.gcp_auth_service_factory import GCPAuthServiceFactory
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset.providers.lazy_entity_factory import LazyEntityFactory
from src.data.dataset.manifests.matching_manifest import MatchingManifest
from src.data.dataset.providers.manifest_instance_provider import ManifestInstanceProvider
from src.data.dataset.registries.suffix_file_registry import SuffixFileRegistry
from src.data.dataset.selectors.determ_string_selector import DetermStringSelector
from src.data.dataset.splitters.string_set_splitter import StringSetSplitter
from src.data.dataset.streams.dock_stream import DockStream
from src.data.dataset.streams.factories.gcs_eval_stream_factory import GCSEvalStreamFactory
from src.data.decoders.factories.darwin_decoder_factory import DarwinDecoderFactory
from src.data.dataset.label import SimpleLabelParserFactory
from src.data.loading.loaders.factories.gcs_loader_factory import GCSLoaderFactory
from src.data.parsing.base_name_parser import BaseNameParser
from src.data.streaming.managers.throttled_streamer_manager import ThrottledStreamerManager
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from src.utils.norsvin_dataset_config import NORSVIN_SPLIT_RATIOS
from tests.utils.gcs.test_bucket import TestBucket
from tests.utils.annotated_frame_visualizer import AnnotatedFrameVisualizer


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
        selector=DetermStringSelector(strings=splitter.splits[2])
    )


@pytest.fixture
def entity_factory(loader_factory):
    """Fixture to provide an EntityFactory instance."""
    return LazyEntityFactory(loader_factory, BaseNameParser())


@pytest.fixture
def stream():
    """Fixture to provide a DockStream instance."""
    return DockStream[AnnotatedFrame]()


@pytest.fixture
def manager(streamer_factory, stream):
    """Fixture to provide a DockingStreamerManager instance."""
    return ThrottledStreamerManager[AnnotatedFrame](
        streamer_factory=streamer_factory,
        stream=stream
    )


def test_streaming_test_set(stream, manager):
    """Tests that the test set stream gives instances deterministically."""
    # arrange
    manager.run()

    # act
    instance = stream.read()
    while instance:
        assert isinstance(instance, AnnotatedFrame)
        instance = stream.read()
        AnnotatedFrameVisualizer.visualize(instance)
    print(f"Finished reading!")
    manager.stop()

def test_norsvin_test_stream():
    """Tests Norsvin test set stream."""
    # arrange
    factory = GCSEvalStreamFactory(
        bucket_name=TestBucket.BUCKET_NAME,
        service_account_path=TestBucket.SERVICE_ACCOUNT_FILE,
        split=2,
        frame_size=(1920, 1080),
        label_map=NorsvinBehaviorClass.get_label_map(),
        split_ratios=NORSVIN_SPLIT_RATIOS,
        buffer_size=3
    )
    managed_stream = factory.create_stream()

    # act
    managed_stream.start()
    stream = managed_stream.stream
    instance = stream.read()
    count = 0
    while instance:
        assert isinstance(instance, AnnotatedFrame)
        instance = stream.read()
        # StreamedAnnotatedFrameVisualizer.visualize(instance)
        count += 1

    print(f"COUNT: {count}")
    managed_stream.stop()