from typing import TypeVar, Generic, Callable, Any, Optional

from src.data.pipeline.base_component import BaseComponent, O
from src.data.pipeline.component import Component
from src.data.pipeline.consumer import Consumer
from src.data.processing.processor import Processor

# input type
I = TypeVar("I")

# field type
P = TypeVar("P")

# processed field type
R = TypeVar("R")


class FieldTransformer(Generic[I, P]):
    """Fluent builder for constructing field-level transformation components."""

    def __init__(self, extractor: Callable[[I], P], field_name: str):
        """
        Initializes a FieldTransformer instance.

        Args:
            extractor (Callable[[I], P]): function that will extract the field
            field_name (str): name of the field to transform
        """
        self._extractor = extractor
        self._field_name = field_name

    @classmethod
    def of(cls, field_name: str) -> "FieldTransformer[I, Any]":
        """
        Creates a transformer for a field by name (uses getattr / setattr).

        Args:
            field_name (str): the name of the field to transform

        Returns:
            FieldTransformer[I, Any]: the field transformer
        """

        def extractor(obj: I) -> Any:
            return getattr(obj, field_name)

        return cls(extractor, field_name)

    def using(self, processor: Processor[P, R]) -> Component[I, I]:
        """
        Applies the processor to the extracted field and injects the result.

        Args:
            processor (processor[P, R]): the processor to apply

        Returns:
            Component[I, I]: a pipeline-ready component
        """
        extractor = self._extractor
        field_name = self._field_name

        class _FieldTransformComponent(BaseComponent[I, I]):

            def __init__(self):
                super().__init__()

            def _consume(self, data: I, consumer: Consumer[O]) -> bool:
                field = extractor(data)
                processed = processor.process(field)

                setattr(data, field_name, processed)

                return consumer.consume(data)

        return _FieldTransformComponent()
