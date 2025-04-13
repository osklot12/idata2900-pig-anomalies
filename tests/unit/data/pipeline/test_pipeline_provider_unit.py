from typing import Optional
from unittest.mock import Mock

import pytest

from src.data.pipeline.component import Component
from src.data.pipeline.component_factory import ComponentFactory
from src.data.pipeline.consumer import Consumer
from src.data.pipeline.consumer_provider import ConsumerProvider
from src.data.pipeline.pipeline_provider import PipelineProvider
from src.data.structures.atomic_bool import AtomicBool


class DummyComponent(Component[str]):

    def __init__(self, suffix: str):
        self.suffix = suffix
        self.consumer = None

    def connect(self, consumer: Consumer[str]) -> None:
        self.consumer = consumer

    def consume(self, data: Optional[str]) -> bool:
        return self.consumer.consume(data + self.suffix)


class DummyConsumerProvider(ConsumerProvider[str]):

    def __init__(self, consumer: Consumer[str]):
        self.consumer = consumer

    def get_consumer(self, release: Optional[AtomicBool] = None) -> Optional[Consumer[str]]:
        return self.consumer


def _create_component_factory(component: Component[str]) -> ComponentFactory[str]:
    """Creates a component factory that returns the given component."""
    factory = Mock()
    factory.create_component.return_value = component
    return factory


@pytest.mark.unit
def test_data_goes_through_all_components_successfully():
    """Tests that data goes through all components of the pipeline successfully."""
    # arrange
    factory1 = _create_component_factory(DummyComponent("a"))
    factory2 = _create_component_factory(DummyComponent("b"))
    factory3 = _create_component_factory(DummyComponent("c"))

    final_consumer = Mock()
    consumer_provider = DummyConsumerProvider(final_consumer)

    pipeline_provider = PipelineProvider(factory1, factory2, factory3,
                                         consumer_provider=consumer_provider)
    pipeline = pipeline_provider.get_consumer()

    # act
    pipeline.consume("")

    # assert
    final_consumer.consume.assert_called_once()
    args, kwargs = final_consumer.consume.call_args
    assert args[0] == "abc"