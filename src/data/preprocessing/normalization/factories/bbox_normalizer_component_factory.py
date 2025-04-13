from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.pipeline.component import Component
from src.data.pipeline.component_factory import ComponentFactory, T
from src.data.preprocessing.normalization.factories.bbox_normalizer_factory import BBoxNormalizerFactory
from src.data.preprocessing.normalization.normalizers.bbox_normalizer_component import BBoxNormalizerComponent


class BBoxNormalizerComponentFactory(ComponentFactory[StreamedAnnotatedFrame]):
    """Factory for creating BBoxNormalizerComponent instances."""

    def __init__(self, normalizer_factory: BBoxNormalizerFactory):
        """
        Initializes a BBoxNormalizerComponentFactory instance.

        Args:
            normalizer_factory (BBoxNormalizerFactory): factory for creating BBoxNormalizer instances
        """
        self._normalizer_factory = normalizer_factory

    def create_component(self) -> Component[StreamedAnnotatedFrame]:
        return BBoxNormalizerComponent(normalizer=self._normalizer_factory.create_bbox_normalizer())