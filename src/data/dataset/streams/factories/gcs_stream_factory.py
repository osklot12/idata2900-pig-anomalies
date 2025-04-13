from typing import TypeVar, Generic, List, Optional, Dict

from src.auth.factories.auth_service_factory import AuthServiceFactory
from src.auth.factories.gcp_auth_service_factory import GCPAuthServiceFactory
from src.data.dataclasses.dataset_split_ratios import DatasetSplitRatios
from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.manifests.manifest import Manifest
from src.data.dataset.manifests.matching_manifest import MatchingManifest
from src.data.dataset.providers.entity_factory import EntityFactory
from src.data.dataset.providers.instance_provider import InstanceProvider
from src.data.dataset.providers.lazy_entity_factory import LazyEntityFactory
from src.data.dataset.providers.manifest_instance_provider import ManifestInstanceProvider
from src.data.dataset.registries.file_registry import FileRegistry
from src.data.dataset.registries.suffix_file_registry import SuffixFileRegistry
from src.data.dataset.selectors.determ_string_selector import DetermStringSelector
from src.data.dataset.selectors.random_string_selector import RandomStringSelector
from src.data.dataset.selectors.selector import Selector
from src.data.dataset.splitters.string_set_splitter import StringSetSplitter
from src.data.dataset.streams.dock_stream import DockStream
from src.data.dataset.streams.factories.managed_stream_factory import ManagedStreamFactory
from src.data.dataset.streams.managed_stream import ManagedStream
from src.data.dataset.streams.pool_stream import PoolStream
from src.data.dataset.streams.writable_stream import WritableStream
from src.data.decoders.factories.annotation_decoder_factory import AnnotationDecoderFactory
from src.data.decoders.factories.darwin_decoder_factory import DarwinDecoderFactory
from src.data.label.factories.label_parser_factory import LabelParserFactory
from src.data.label.factories.simple_label_parser_factory import SimpleLabelParserFactory
from src.data.loading.factories.gcs_loader_factory import GCSLoaderFactory
from src.data.loading.factories.loader_factory import LoaderFactory
from src.data.parsing.base_name_parser import BaseNameParser
from src.data.pipeline.component_factory import ComponentFactory
from src.data.pipeline.consumer_provider import ConsumerProvider
from src.data.pipeline.pipeline_provider import PipelineProvider
from src.data.streaming.managers.stream_feeding_manager import StreamFeedingManager
from src.data.streaming.managers.streamer_manager import StreamerManager
from src.data.streaming.streamers.factories.instance_streamer_factory import InstanceStreamerFactory
from src.data.streaming.streamers.factories.streamer_factory import StreamerFactory
from src.typevars.enum_type import T_Enum
from src.utils.gcs_credentials import GCSCredentials

T = TypeVar("T")


class GCSStreamFactory(Generic[T], ManagedStreamFactory[T]):
    """Factory for creating managed Google Cloud Storage (GCS) streams."""

    def __init__(self, gcs_creds: GCSCredentials,
                 split_ratios: DatasetSplitRatios,
                 split: DatasetSplit,
                 label_map: Dict[str, T_Enum],
                 preprocessor_factories: Optional[List[ComponentFactory[T]]] = None):
        """
        Initializes a GCSStreamFactory instance.

        Args:
            gcs_creds (GCSCredentials): Google Cloud Storage credentials
            split_ratios (DatasetSplitRatios): dataset split ratios
            split (DatasetSplit): dataset split to create stream for
            label_map (Dict[str, T_Enum]): label map for annotation classes
            preprocessor_factories (Optional[List[ComponentFactory[T]]]): optional list of preprocessor factories to use, defaults to None
        """
        self._gcs_creds = gcs_creds
        self._split_ratios = split_ratios
        self._split = split
        self._label_map = label_map
        self._preprocessor_factories = preprocessor_factories if preprocessor_factories is not None else []

    def create_stream(self) -> ManagedStream[T]:
        auth_factory = self._create_auth_service_factory(self._gcs_creds.service_account_path)
        label_parser_factory = self._create_label_parser_factory(self._label_map)
        decoder_factory = self._create_decoder_factory(label_parser_factory)
        loader_factory = self._create_loader_factory(self._gcs_creds.bucket_name, auth_factory, decoder_factory)

        manifest = self._create_manifest(loader_factory.create_file_registry())
        splitter = self._create_splitter(manifest.ids, self._split_ratios)
        selector = self._create_selector(splitter.splits, self._split)
        instance_provider = self._create_instance_provider(manifest, selector)
        entity_provider = self._create_entity_provider(loader_factory)
        streamer_factory = self._create_streamer_factory(instance_provider, entity_provider)

        stream = self._create_stream(self._split)
        pipeline_provider = PipelineProvider(*self._preprocessor_factories, consumer_provider=stream)
        manager = self._create_streamer_manager(streamer_factory, pipeline_provider)

        return ManagedStream[T](stream=stream, manager=manager)


    @staticmethod
    def _create_auth_service_factory(service_account_path: str) -> AuthServiceFactory:
        """Creates an AuthServiceFactory instance."""
        print(f"[GCSTrainingStreamFactory] Created auth service")
        return GCPAuthServiceFactory(service_account_path)

    @staticmethod
    def _create_label_parser_factory(label_map: Dict[str, T_Enum]) -> LabelParserFactory:
        """Creates a LabelParserFactory instance."""
        print(f"[GCSTrainingStreamFactory] Created SimpleLabelParserFactory")
        return SimpleLabelParserFactory(label_map)

    @staticmethod
    def _create_decoder_factory(label_parser_factory: LabelParserFactory) -> AnnotationDecoderFactory:
        """Creates an AnnotationDecoderFactory instance."""
        print(f"[GCSTrainingStreamFactory] Created DarwinDecoderFactory")
        return DarwinDecoderFactory(label_parser_factory)

    @staticmethod
    def _create_loader_factory(bucket_name: str, auth_service_factory: AuthServiceFactory,
                               decoder_factory: AnnotationDecoderFactory) -> LoaderFactory:
        """Creates a LoaderFactory instance."""
        print(f"[GCSTrainingStreamFactory] Created LoaderFactory")
        return GCSLoaderFactory(
            bucket_name=bucket_name,
            auth_factory=auth_service_factory,
            decoder_factory=decoder_factory,
        )

    @staticmethod
    def _create_manifest(source: FileRegistry) -> Manifest:
        """Creates a dataset manifest."""
        print(f"[GCSTrainingStreamFactory] Created MatchinManifest")
        return MatchingManifest(
            video_registry=SuffixFileRegistry(source=source, suffixes=("mp4",)),
            annotations_registry=SuffixFileRegistry(source=source, suffixes=("json",)),
        )

    @staticmethod
    def _create_splitter(ids: List[str], split_ratios: DatasetSplitRatios) -> StringSetSplitter:
        """Creates a DetermSplitter instance."""
        print(f"[GCSTrainingStreamFactory] Created Splitter")
        return StringSetSplitter(
            strings=ids,
            weights=[
                split_ratios.train,
                split_ratios.val,
                split_ratios.test
            ]
        )

    @staticmethod
    def _create_selector(splits: List[List[str]], split: DatasetSplit) -> Selector[str]:
        """Creates a selector for selecting dataset instances."""
        print(f"[GCSTrainingStreamFactory] Created Selector")
        if split == DatasetSplit.TRAIN:
            selector = RandomStringSelector(strings=splits[0])
        elif split == DatasetSplit.VAL:
            selector = DetermStringSelector(strings=splits[1])
        else:
            selector = DetermStringSelector(strings=splits[2])

        return selector

    @staticmethod
    def _create_instance_provider(manifest: Manifest, selector: Selector[str]) -> InstanceProvider:
        """Creates a InstanceProvider instance."""
        print(f"[GCSTrainingStreamFactory] Created InstanceProvider")
        return ManifestInstanceProvider(manifest=manifest, selector=selector)

    @staticmethod
    def _create_entity_provider(loader_factory: LoaderFactory) -> EntityFactory:
        """Creates a DatasetEntityProvider instance."""
        print(f"[GCSTrainingStreamFactory] Created EntityProvider")
        return LazyEntityFactory(
            loader_factory=loader_factory,
            id_parser=BaseNameParser()
        )

    @staticmethod
    def _create_streamer_factory(instance_provider: InstanceProvider, entity_factory: EntityFactory) -> StreamerFactory[T]:
        """Creates an AggregatedStreamerFactory instance."""
        print(f"[GCSTrainingStreamFactory] Created StreamerFactory")
        return InstanceStreamerFactory(
            instance_provider=instance_provider,
            entity_factory=entity_factory
        )

    @staticmethod
    def _create_stream(split: DatasetSplit) -> WritableStream[T]:
        """Creates a Stream instance."""
        if split == DatasetSplit.TRAIN:
            stream = PoolStream(pool_size=2000)
        else:
            stream = DockStream(buffer_size=3, dock_size=300)

        return stream

    @staticmethod
    def _create_streamer_manager(streamer_factory: StreamerFactory[T], consumer_provider: ConsumerProvider[T]) -> StreamerManager:
        """Creates a StreamerManager instance."""
        return StreamFeedingManager(
            streamer_factory=streamer_factory,
            provider=consumer_provider,
            max_streamers=2
        )