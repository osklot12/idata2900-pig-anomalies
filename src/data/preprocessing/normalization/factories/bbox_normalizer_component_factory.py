from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.pipeline.component import Component
from src.data.pipeline.component_factory import ComponentFactory
from src.data.preprocessing.normalization.factories.bbox_normalizer_factory import BBoxNormalizerFactory
from src.data.preprocessing.normalization.normalizers.bbox_normalizer_component import BBoxNormalizerComponent


class BBoxNormalizerComponentFactory(ComponentFactory[AnnotatedFrame, AnnotatedFrame]):
    """Factory for creating BBoxNormalizerComponent instances."""

    def __init__(self, normalizer_factory: BBoxNormalizerFactory):
        """
        Initializes a BBoxNormalizerComponentFactory instance.

        Args:
            normalizer_factory (BBoxNormalizerFactory): factory for creating BBoxNormalizer instances
        """
        self._normalizer_factory = normalizer_factory

    def create_component(self) -> Component[AnnotatedFrame, AnnotatedFrame]:
        return BBoxNormalizerComponent(normalizer=self._normalizer_factory.create_bbox_normalizer())