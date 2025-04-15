from typing import Dict

from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.pipeline.component import Component
from src.data.pipeline.component_factory import ComponentFactory, T
from src.data.preprocessing.class_balancer import ClassBalancer
from src.typevars.enum_type import T_Enum


class ClassBalancerFactory(ComponentFactory[StreamedAnnotatedFrame]):
    """Factory for creating ClassBalancer instances."""

    def __init__(self, class_counts: Dict[T_Enum, int], max_samples_per: int):
        """
        Initializes a ClassBalancerFactory instance.

        Args:
           class_counts (Dict[T_Enum, int]): a dictionary containing the count of each class
           max_samples_per (int): the maximum number of copies allowed per instance
        """
        self._class_counts = class_counts
        self._max_samples_per = max_samples_per


    def create_component(self) -> Component[T]:
        return ClassBalancer(
            class_counts=dict(self._class_counts),
            max_samples_per=self._max_samples_per
        )