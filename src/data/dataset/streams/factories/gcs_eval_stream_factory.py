from typing import List, Tuple, Dict

from src.auth.factories.auth_service_factory import AuthServiceFactory
from src.auth.factories.gcp_auth_service_factory import GCPAuthServiceFactory
from src.data.dataclasses.dataset_split_ratios import DatasetSplitRatios
from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.dataset.manifests.manifest import Manifest
from src.data.dataset.manifests.matching_manifest import MatchingManifest
from src.data.dataset.providers.entity_factory import EntityFactory
from src.data.dataset.providers.instance_provider import InstanceProvider
from src.data.dataset.providers.lazy_entity_factory import LazyEntityFactory
from src.data.dataset.providers.manifest_instance_provider import ManifestInstanceProvider
from src.data.dataset.registries.file_registry import FileRegistry
from src.data.dataset.registries.suffix_file_registry import SuffixFileRegistry
from src.data.dataset.selectors.determ_string_selector import DetermStringSelector
from src.data.dataset.splitters.string_set_splitter import StringSetSplitter
from src.data.dataset.streams.dock_stream import DockStream
from src.data.dataset.streams.factories.managed_stream_factory import ManagedStreamFactory, T
from src.data.dataset.streams.managed_stream import ManagedStream
from src.data.decoders.factories.annotation_decoder_factory import AnnotationDecoderFactory
from src.data.decoders.factories.darwin_decoder_factory import DarwinDecoderFactory
from src.data.label.factories.label_parser_factory import LabelParserFactory
from src.data.label.factories.simple_label_parser_factory import SimpleLabelParserFactory
from src.data.loading.factories.gcs_loader_factory import GCSLoaderFactory
from src.data.loading.factories.loader_factory import LoaderFactory
from src.data.parsing.base_name_parser import BaseNameParser
from src.data.preprocessing.normalization.factories.simple_bbox_normalizer_factory import SimpleBBoxNormalizerFactory
from src.data.preprocessing.resizing.factories.static_frame_resizer_factory import StaticFrameResizerFactory
from src.data.streaming.managers.stream_feeding_manager import StreamFeedingManager
from src.data.streaming.managers.streamer_manager import StreamerManager
from src.data.streaming.streamers.providers.aggregated_streamer_provider import AggregatedStreamerProvider
from src.data.streaming.streamers.providers.file_streamer_pair_provider import FileStreamerPairProvider
from src.data.streaming.streamers.providers.streamer_pair_provider import StreamerPairProvider
from src.data.streaming.streamers.providers.streamer_factory import StreamerFactory
from src.typevars.enum_type import T_Enum

class GCSEvalStreamFactory(ManagedStreamFactory[StreamedAnnotatedFrame]):

    def __init__(self, bucket_name: str, service_account_path: str, split: int, frame_size: Tuple[int, int],
                 label_map: Dict[str, T_Enum], split_ratios: DatasetSplitRatios, buffer_size: int):
        """
        Initializes a NorsvinTrainingStreamFactory.

        Args:
            service_account_path (str): the path to the service account json file
        """
        self._bucket_name = bucket_name
        self._service_account_path = service_account_path
        self._split = split
        self._frame_size = frame_size
        self._label_map = label_map
        self._split_ratios = split_ratios
        self._buffer_size = buffer_size

    def create_stream(self) -> ManagedStream[StreamedAnnotatedFrame]:
        auth_service_factory = self._create_auth_service_factory(self._service_account_path)
        label_parser_factory = self._create_label_parser_factory(self._label_map)
        decoder_factory = self._create_decoder_factory(label_parser_factory)
        loader_factory = self._create_loader_factory(self._bucket_name, auth_service_factory, decoder_factory)

        manifest = self._create_manifest(loader_factory.create_file_registry())
        splitter = self._create_splitter(manifest.ids, self._split_ratios)
        instance_provider = self._create_instance_provider(manifest, splitter, self._split)

        entity_provider = self._create_entity_provider(loader_factory)
        streamer_pair_provider = self._create_streamer_pair_provider(instance_provider, entity_provider, self._frame_size)
        streamer_factory = self._create_streamer_factory(streamer_pair_provider)

        stream = self._create_stream(self._buffer_size)
        streamer_manager = self._create_streamer_manager(streamer_factory, stream)

        return ManagedStream(stream, streamer_manager)


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

    @staticmethod
    def _create_instance_provider(manifest: Manifest, splitter: StringSetSplitter, split: int) -> InstanceProvider:
        """Creates a InstanceProvider instance."""
        return ManifestInstanceProvider(
            manifest=manifest,
            selector=DetermStringSelector(strings=splitter.splits[split])
        )

    @staticmethod
    def _create_entity_provider(loader_factory: LoaderFactory) -> EntityFactory:
        """Creates a DatasetEntityProvider instance."""
        return LazyEntityFactory(
            loader_factory=loader_factory,
            id_parser=BaseNameParser()
        )

    @staticmethod
    def _create_streamer_pair_provider(instance_provider, entity_factory, frame_size: Tuple[int, int]) -> StreamerPairProvider:
        """Creates a StreamerPairProvider instance."""
        return FileStreamerPairProvider(
            instance_provider=instance_provider,
            entity_factory=entity_factory,
            frame_resizer_factory=StaticFrameResizerFactory(frame_size),
            bbox_normalizer_factory=SimpleBBoxNormalizerFactory((0, 1))
        )

    @staticmethod
    def _create_streamer_factory(streamer_pair_provider: StreamerPairProvider) -> StreamerFactory:
        """Creates an AggregatedStreamerFactory instance."""
        return AggregatedStreamerProvider(streamer_pair_provider=streamer_pair_provider)

    @staticmethod
    def _create_stream(pool_size: int) -> DockStream[StreamedAnnotatedFrame]:
        """Creates a PoolStream instance."""
        return DockStream(buffer_size=pool_size)

    @staticmethod
    def _create_streamer_manager(streamer_factory: StreamerFactory[StreamedAnnotatedFrame],
                                 stream: DockStream[StreamedAnnotatedFrame]) -> StreamerManager:
        """Creates a StaticStreamerManager instance."""
        return StreamFeedingManager(
            streamer_factory=streamer_factory,
            stream=stream
        )