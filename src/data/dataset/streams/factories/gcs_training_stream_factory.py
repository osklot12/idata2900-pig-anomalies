from typing import List, Tuple, Iterable, Dict

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
from src.data.dataset.selectors.random_string_selector import RandomStringSelector
from src.data.dataset.splitters.determ_splitter import DetermSplitter
from src.data.dataset.streams.factories.managed_stream_factory import ManagedStreamFactory, T
from src.data.dataset.streams.managed_stream import ManagedStream
from src.data.dataset.streams.pool_stream import PoolStream
from src.data.decoders.factories.annotation_decoder_factory import AnnotationDecoderFactory
from src.data.decoders.factories.darwin_decoder_factory import DarwinDecoderFactory
from src.data.label.factories.label_parser_factory import LabelParserFactory
from src.data.label.factories.simple_label_parser_factory import SimpleLabelParserFactory
from src.data.loading.factories.gcs_loader_factory import GCSLoaderFactory
from src.data.loading.factories.loader_factory import LoaderFactory
from src.data.parsing.base_name_parser import BaseNameParser
from src.data.preprocessing.augmentation.augmentors.annotated_frame_augmentor import AnnotatedFrameAugmentor
from src.data.preprocessing.augmentation.augmentors.cond_augmentor import CondMultiplier
from src.data.preprocessing.normalization.factories.simple_bbox_normalizer_factory import SimpleBBoxNormalizerFactory
from src.data.preprocessing.preprocessor import Preprocessor
from src.data.preprocessing.resizing.factories.static_frame_resizer_factory import StaticFrameResizerFactory
from src.data.streaming.managers.static_streamer_manager import StaticStreamerManager
from src.data.streaming.managers.streamer_manager import StreamerManager
from src.data.streaming.streamers.providers.aggregated_streamer_provider import AggregatedStreamerProvider
from src.data.streaming.streamers.providers.file_streamer_pair_provider import FileStreamerPairProvider
from src.data.streaming.streamers.providers.streamer_pair_provider import StreamerPairProvider
from src.data.streaming.streamers.providers.streamer_factory import StreamerFactory
from src.typevars.enum_type import T_Enum


class GCSTrainingStreamFactory(ManagedStreamFactory[StreamedAnnotatedFrame]):
    """Factory for creating Norsvin training set streams."""

    def __init__(self, bucket_name: str, service_account_path: str, frame_size: Tuple[int, int],
                 label_map: Dict[str, T_Enum], split_ratios: DatasetSplitRatios, pool_size: int):
        """
        Initializes a NorsvinTrainingStreamFactory.

        Args:
            service_account_path (str): the path to the service account json file
        """
        self._bucket_name = bucket_name
        self._service_account_path = service_account_path
        self._frame_size = frame_size
        self._label_map = label_map
        self._split_ratios = split_ratios
        self._pool_size = pool_size

    def create_stream(self) -> ManagedStream[StreamedAnnotatedFrame]:
        auth_service_factory = self._create_auth_service_factory(self._service_account_path)
        label_parser_factory = self._create_label_parser_factory(self._label_map)
        decoder_factory = self._create_decoder_factory(label_parser_factory)
        loader_factory = self._create_loader_factory(self._bucket_name, auth_service_factory, decoder_factory)

        manifest = self._create_manifest(loader_factory.create_file_registry())
        splitter = self._create_splitter(manifest.ids, self._split_ratios)
        instance_provider = self._create_instance_provider(manifest, splitter)

        entity_provider = self._create_entity_provider(loader_factory)
        streamer_pair_provider = self._create_streamer_pair_provider(instance_provider, entity_provider, self._frame_size)
        streamer_factory = self._create_streamer_factory(streamer_pair_provider)

        instance_multiplier = self._create_instance_multiplier()
        augmentor = self._create_augmentor()
        stream = self._create_stream(self._pool_size, [instance_multiplier, augmentor])
        streamer_manager = self._create_streamer_manager(streamer_factory, stream)

        print(f"[GCSTrainingStreamFactory] Returned ManagedStream")
        return ManagedStream(stream, streamer_manager)


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
    def _create_splitter(ids: List[str], split_ratios: DatasetSplitRatios) -> DetermSplitter:
        """Creates a DetermSplitter instance."""
        print(f"[GCSTrainingStreamFactory] Created Splitter")
        return DetermSplitter(
            strings=ids,
            weights=[
                split_ratios.train,
                split_ratios.val,
                split_ratios.test
            ]
        )

    @staticmethod
    def _create_instance_provider(manifest: Manifest, splitter: DetermSplitter) -> InstanceProvider:
        """Creates a InstanceProvider instance."""
        print(f"[GCSTrainingStreamFactory] Created InstanceProvider")
        return ManifestInstanceProvider(
            manifest=manifest,
            selector=RandomStringSelector(strings=splitter.splits[0])
        )

    @staticmethod
    def _create_entity_provider(loader_factory: LoaderFactory) -> EntityFactory:
        """Creates a DatasetEntityProvider instance."""
        print(f"[GCSTrainingStreamFactory] Created EntityProvider")
        return LazyEntityFactory(
            loader_factory=loader_factory,
            id_parser=BaseNameParser()
        )

    @staticmethod
    def _create_streamer_pair_provider(instance_provider, entity_factory, frame_size: Tuple[int, int]) -> StreamerPairProvider:
        """Creates a StreamerPairProvider instance."""
        print(f"[GCSTrainingStreamFactory] Created StreamerPairProvider")
        return FileStreamerPairProvider(
            instance_provider=instance_provider,
            entity_factory=entity_factory,
            frame_resizer_factory=StaticFrameResizerFactory(frame_size),
            bbox_normalizer_factory=SimpleBBoxNormalizerFactory((0, 1))
        )

    @staticmethod
    def _create_streamer_factory(streamer_pair_provider: StreamerPairProvider) -> StreamerFactory:
        """Creates an AggregatedStreamerFactory instance."""
        print(f"[GCSTrainingStreamFactory] Created StreamerFactory")
        return AggregatedStreamerProvider(streamer_pair_provider=streamer_pair_provider)

    @staticmethod
    def _create_instance_multiplier():
        """Creates an CondMultiplier instance."""
        def is_annotated(instance: StreamedAnnotatedFrame) -> bool:
            return len(instance.annotations) > 0

        return CondMultiplier(10, is_annotated)

    @staticmethod
    def _create_augmentor() -> Preprocessor:
        """Creates an AnnotatedFrameAugmentor instance."""
        return AnnotatedFrameAugmentor()

    @staticmethod
    def _create_stream(pool_size: int, preprocessors: Iterable[Preprocessor]) -> PoolStream[StreamedAnnotatedFrame]:
        """Creates a PoolStream instance."""
        print(f"[GCSTrainingStreamFactory] Created Stream")
        return PoolStream[StreamedAnnotatedFrame](
            pool_size=pool_size,
            preprocessors=[preprocessor for preprocessor in preprocessors]
        )

    @staticmethod
    def _create_streamer_manager(streamer_factory: StreamerFactory[StreamedAnnotatedFrame],
                                 stream: PoolStream[StreamedAnnotatedFrame]) -> StreamerManager:
        """Creates a StaticStreamerManager instance."""
        print(f"[GCSTrainingStreamFactory] Created StreamManager")
        return StaticStreamerManager[StreamedAnnotatedFrame](
            streamer_factory=streamer_factory,
            consumer=stream,
            n_streamers=2
        )