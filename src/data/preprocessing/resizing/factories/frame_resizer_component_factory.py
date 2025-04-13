from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.pipeline.component import Component
from src.data.pipeline.component_factory import ComponentFactory, T
from src.data.preprocessing.resizing.factories.frame_resizer_factory import FrameResizerFactory
from src.data.preprocessing.resizing.resizers.frame_resizer_component import FrameResizerComponent


class FrameResizerComponentFactory(ComponentFactory[StreamedAnnotatedFrame]):
    """Factory for creating FrameResizerComponent instances."""

    def __init__(self, resizer_factory: FrameResizerFactory):
        """
        Initializes a FrameResizerComponentFactory instance.

        Args:
            resizer_factory (FrameResizerFactory): factory for creating frame resizer instances
        """
        self._resizer_factory = resizer_factory

    def create_component(self) -> Component[StreamedAnnotatedFrame]:
        return FrameResizerComponent(resizer=self._resizer_factory.create_frame_resizer())