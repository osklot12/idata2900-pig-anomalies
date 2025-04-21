from typing import TypeVar, Optional

from src.data.pipeline.consumer import Consumer
from src.data.pipeline.consumer_provider import ConsumerProvider
from src.data.pipeline.factories.pipeline_factory import PipelineFactory
from src.data.structures.atomic_bool import AtomicBool

# input data type for pipeline
I = TypeVar("I")

# output data type for pipeline and input type for sink
O = TypeVar("O")


class PipelineToSinkProvider(ConsumerProvider[I]):
    """Provider of complete pipelines."""

    def __init__(self, pipeline_factory: PipelineFactory[I, O], sink_provider: ConsumerProvider[O]):
        """
        Initializes a PipelineToSinkProvider instance.

        Args:
            pipeline_factory (PipelineFactory[I, O]): factory for creating pipelines
            sink_provider (ConsumerProvider[O]): factory for creating data sinks
        """
        self._pipeline_factory = pipeline_factory
        self._sink_provider = sink_provider

    def get_consumer(self, release: Optional[AtomicBool] = None) -> Optional[Consumer[I]]:
        consumer = None

        sink = self._sink_provider.get_consumer(release=release)
        if sink is not None:
            pipeline = self._pipeline_factory.create_pipeline()
            consumer = pipeline.into(sink)

        return consumer