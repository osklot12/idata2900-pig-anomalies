from src.data.dataset.providers.entity_factory import EntityFactory
from src.data.dataset.providers.factories.entity_provider_factory import EntityProviderFactory
from src.data.dataset.providers.factories.instance_provider_factory import InstanceProviderFactory
from src.data.preprocessing.normalization.factories.bbox_normalizer_factory import BBoxNormalizerFactory
from src.data.preprocessing.resizing.factories.frame_resizer_factory import FrameResizerFactory
from src.data.streaming.streamers.providers.factories.streamer_pair_provider_factory import StreamerPairProviderFactory
from src.data.streaming.streamers.providers.file_streamer_pair_provider import FileStreamerPairProvider
from src.data.streaming.streamers.providers.streamer_pair_provider import StreamerPairProvider


class FileStreamerPairProviderFactory(StreamerPairProviderFactory):
    """Factory for creating streamer pair provider factories."""

    def __init__(self, instance_provider_factory: InstanceProviderFactory, entity_provider_factory: EntityProviderFactory,
                 frame_resizer_factory: FrameResizerFactory, bbox_normalizer_factory: BBoxNormalizerFactory):
        """
        Initializes a FileStreamerPairProviderFactory instance.

        Args:
            instance_provider_factory (InstanceProviderFactory): factory for creating instance providers
            entity_provider_factory (EntityProviderFactory): factory for creating entity providers
            frame_resizer_factory (FrameResizerFactory): factory for creating frame resizers
            bbox_normalizer_factory (BBoxNormalizerFactory): factory for creating bounding box normalizers
        """
        self._instance_provider_factory = instance_provider_factory
        self._entity_provider_factory = entity_provider_factory
        self._frame_resizer_factory = frame_resizer_factory
        self._bbox_normalizer_factory = bbox_normalizer_factory

    def create_pair_provider(self) -> StreamerPairProvider:
        return FileStreamerPairProvider(
            instance_provider=self._instance_provider_factory.create_provider(),
            entity_factory=self._entity_provider_factory.create_provider(),
            frame_resizer_factory=self._frame_resizer_factory,
            bbox_normalizer_factory=self._bbox_normalizer_factory
        )