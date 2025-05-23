    from typing import TypeVar, Generic, List, Dict, Iterable, Optional, Callable
    
    from src.auth.factories.auth_service_factory import AuthServiceFactory
    from src.auth.factories.gcp_auth_service_factory import GCPAuthServiceFactory
    from src.data.dataclasses.dataset_split_ratios import DatasetSplitRatios
    from src.data.dataset.dataset_split import DatasetSplit
    from src.data.dataset.label.factories.simple_label_parser_factory import SimpleLabelParserFactory
    from src.data.dataset.manifests.manifest import Manifest
    from src.data.dataset.manifests.matching_manifest import MatchingManifest
    from src.data.dataset.metamakers.file_metamaker import FileMetamaker
    from src.data.dataset.providers.entity_factory import EntityFactory
    from src.data.dataset.providers.instance_provider import InstanceProvider
    from src.data.dataset.providers.lazy_entity_factory import LazyEntityFactory
    from src.data.dataset.providers.manifest_instance_provider import ManifestInstanceProvider
    from src.data.dataset.registries.file_registry import FileRegistry
    from src.data.dataset.registries.suffix_file_registry import SuffixFileRegistry
    from src.data.dataset.selectors.factories.selector_factory import SelectorFactory
    from src.data.dataset.selectors.selector import Selector
    from src.data.dataset.splitters.factories.string_set_splitter_factory import StringSetSplitterFactory
    from src.data.dataset.splitters.string_set_splitter import StringSetSplitter
    from src.data.dataset.streams.closable import Closable
    from src.data.dataset.streams.factories.writable_stream_factory import WritableStreamFactory
    from src.data.dataset.streams.managed.factories.manged_stream_factory import ManagedStreamFactory
    from src.data.dataset.streams.managed.managed_stream import ManagedStream
    from src.data.decoders.factories.annotation_decoder_factory import AnnotationDecoderFactory
    from src.data.decoders.factories.darwin_decoder_factory import DarwinDecoderFactory
    from src.data.dataset.label.factories.label_parser_factory import LabelParserFactory
    from src.data.loading.loaders.factories.gcs_loader_factory import GCSLoaderFactory
    from src.data.loading.loaders.factories.loader_factory import LoaderFactory
    from src.data.parsing.base_name_parser import BaseNameParser
    from src.data.pipeline.consumer_provider import ConsumerProvider
    from src.data.pipeline.factories.pipeline_factory import PipelineFactory
    from src.data.pipeline.pipeline_to_sink_provider import PipelineToSinkProvider
    from src.data.streaming.managers.throttled_streamer_manager import ThrottledStreamerManager
    from src.data.streaming.managers.streamer_manager import StreamerManager
    from src.data.streaming.streamers.factories.file_streamer_factory import FileStreamerFactory
    from src.data.streaming.streamers.factories.streamer_factory import StreamerFactory
    from src.data.typevars.enum_type import T_Enum
    from src.utils.gcs_credentials import GCSCredentials
    
    # data type read from the stream
    T = TypeVar("T")
    
    # data type fed into pipeline
    A = TypeVar("A")
    
    # data type fed into stream
    B = TypeVar("B")
    
    
    class GCSStreamFactory(Generic[T, A, B], ManagedStreamFactory[T]):
        """Factory for creating managed Google Cloud Storage (GCS) streams."""
    
        def __init__(self, gcs_creds: GCSCredentials,
                     split_ratios: DatasetSplitRatios,
                     split: DatasetSplit,
                     selector_factory: SelectorFactory[str],
                     label_map: Dict[str, T_Enum],
                     stream_factory: WritableStreamFactory[B],
                     pipeline_factory: Optional[PipelineFactory[A, B]] = None,
                     filter_func: Optional[Callable[[Dict[str, int]], bool]] = None,
                     meta_cache_dir: str = "cache/metadata.json"
                     ):
            """
            Initializes a GCSStreamFactory instance.
    
            Args:
                gcs_creds (GCSCredentials): Google Cloud Storage credentials
                split_ratios (DatasetSplitRatios): dataset split ratios
                split (DatasetSplit): dataset split to create stream for
                selector_factory (SelectorFactory[str]): factory for creating selectors of dataset instances
                label_map (Dict[str, T_Enum]): label map for annotation classes
                stream_factory (WritableStreamFactory[B]): factory for creating stream instances
                pipeline_factory (Optional[PipelineFactory[T]]): optional pipeline provider
                filter_func (Optional[Callable[[Dict[str, int]], bool]]): optional filter function
                meta_cache_dir (Optional[str]): cache directory for metadata
            """
            self._gcs_creds = gcs_creds
            self._split_ratios = split_ratios
            self._split = split
            self._selector_factory = selector_factory
            self._label_map = label_map
            self._stream_factory = stream_factory
            self._pipeline_factory: Optional[PipelineFactory[A, B]] = pipeline_factory
            self._filter_func = filter_func
            self._meta_cache_dir = meta_cache_dir
    
        def create_stream(self) -> ManagedStream[T]:
            auth_factory = self._create_auth_service_factory(self._gcs_creds.service_account_path)
            label_parser_factory = self._create_label_parser_factory(self._label_map)
            decoder_factory = self._create_decoder_factory(label_parser_factory)
            loader_factory = self._create_loader_factory(self._gcs_creds.bucket_name, auth_factory, decoder_factory)
    
            ratios = self._split_ratios
            splitter_factory = StringSetSplitterFactory(weights=[ratios.train, ratios.val, ratios.test])
            metamaker = FileMetamaker(
                loader_factory=loader_factory,
                splitter_factory=splitter_factory,
                cache=True,
                cache_dir=self._meta_cache_dir
            )
            metadata = metamaker.make_metadata()
    
            manifest = self._create_manifest(loader_factory.create_file_registry())
            selector = self._create_selector(list(metadata[self._split.value].keys()))
            instance_provider = self._create_instance_provider(manifest, selector)
            entity_provider = self._create_entity_provider(loader_factory)
            streamer_factory = self._create_streamer_factory(instance_provider, entity_provider)
    
            stream = self._stream_factory.create_stream()
            if self._pipeline_factory is not None:
                consumer_provider = PipelineToSinkProvider(
                    pipeline_factory=self._pipeline_factory,
                    sink_provider=stream
                )
                manager = self._create_streamer_manager(streamer_factory, consumer_provider, [stream])
            else:
                manager = self._create_streamer_manager(streamer_factory, stream, [stream])
    
            return ManagedStream[T](stream=stream, manager=manager)
    
        @staticmethod
        def has_annotations(data: Dict[str, int]) -> bool:
            return any(count > 0 for count in data.values())
    
        @staticmethod
        def _create_auth_service_factory(service_account_path: str) -> AuthServiceFactory:
            """Creates an AuthServiceFactory instance."""
            return GCPAuthServiceFactory(service_account_path)
    
        @staticmethod
        def _create_label_parser_factory(label_map: Dict[str, T_Enum]) -> LabelParserFactory:
            """Creates a LabelParserFactory instance."""
            return SimpleLabelParserFactory(label_map)
    
        @staticmethod
        def _create_decoder_factory(label_parser_factory: LabelParserFactory) -> AnnotationDecoderFactory:
            """Creates an AnnotationDecoderFactory instance."""
            return DarwinDecoderFactory(label_parser_factory)
    
        @staticmethod
        def _create_loader_factory(bucket_name: str, auth_service_factory: AuthServiceFactory,
                                   decoder_factory: AnnotationDecoderFactory) -> LoaderFactory:
            """Creates a LoaderFactory instance."""
            return GCSLoaderFactory(
                bucket_name=bucket_name,
                auth_factory=auth_service_factory,
                decoder_factory=decoder_factory,
            )
    
        @staticmethod
        def _create_manifest(source: FileRegistry) -> Manifest:
            """Creates a dataset manifest."""
            return MatchingManifest(
                video_registry=SuffixFileRegistry(source=source, suffixes=("mp4",)),
                annotations_registry=SuffixFileRegistry(source=source, suffixes=("json",)),
            )
    
        @staticmethod
        def _create_splitter(ids: List[str], split_ratios: DatasetSplitRatios) -> StringSetSplitter:
            """Creates a DetermSplitter instance."""
            return StringSetSplitter(
                strings=ids,
                weights=[
                    split_ratios.train,
                    split_ratios.val,
                    split_ratios.test
                ]
            )
    
        def _create_selector(self, candidates: List[str]) -> Selector[str]:
            """Creates a selector for selecting dataset instances."""
            return self._selector_factory.create_selector(candidates=candidates)
    
        @staticmethod
        def _create_instance_provider(manifest: Manifest, selector: Selector[str]) -> InstanceProvider:
            """Creates a InstanceProvider instance."""
            return ManifestInstanceProvider(manifest=manifest, selector=selector)
    
        @staticmethod
        def _create_entity_provider(loader_factory: LoaderFactory) -> EntityFactory:
            """Creates a DatasetEntityProvider instance."""
            return LazyEntityFactory(
                loader_factory=loader_factory,
                id_parser=BaseNameParser()
            )
    
        @staticmethod
        def _create_streamer_factory(instance_provider: InstanceProvider,
                                     entity_factory: EntityFactory) -> FileStreamerFactory:
            """Creates an AggregatedStreamerFactory instance."""
            return FileStreamerFactory(
                instance_provider=instance_provider,
                entity_factory=entity_factory
            )
    
        @staticmethod
        def _create_streamer_manager(streamer_factory: StreamerFactory[T], consumer_provider: ConsumerProvider[T],
                                     closables: Iterable[Closable]) -> StreamerManager:
            """Creates a StreamerManager instance."""
            return ThrottledStreamerManager(
                streamer_factory=streamer_factory,
                provider=consumer_provider,
                closables=closables,
                max_streamers=4
            )