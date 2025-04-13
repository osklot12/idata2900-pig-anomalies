from typing import TypeVar, Generic, Optional

from src.data.pipeline.component_factory import ComponentFactory
from src.data.pipeline.consumer import Consumer
from src.data.pipeline.consumer_provider import ConsumerProvider
from src.data.structures.atomic_bool import AtomicBool

T = TypeVar("T")

class PipelineProvider(Generic[T], ConsumerProvider[T]):
    """Provider of chains of pipeline components."""

    def __init__(self, *component_factories: ComponentFactory[T], consumer_provider: ConsumerProvider[T]):
        """
        Initializes a PipelineProvider instance.

        Args:
            *component_factories (ComponentFactory[T]): factories for creating pipeline components, in the order they are given
            consumer_provider (ConsumerProvider[T]): provider of end consumers of the pipeline
        """
        self._component_factories = component_factories
        self._consumer_provider = consumer_provider

    def get_consumer(self, release: Optional[AtomicBool] = None) -> Optional[Consumer[T]]:
        result = None

        final_consumer = self._consumer_provider.get_consumer(release=release)
        if final_consumer is not None:
            result = final_consumer
            if self._component_factories:
                components = [factory.create_component() for factory in self._component_factories]
                for i in range(len(components) - 1):
                    components[i].connect(components[i + 1])

                components[-1].connect(final_consumer)
                result = components[0]

        return result