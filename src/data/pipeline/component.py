from abc import ABC
from typing import TypeVar, Generic

from src.data.pipeline.consumer import Consumer
from src.data.pipeline.producer import Producer

# input data type
I = TypeVar("I")

# output data type
O = TypeVar("O")


class Component(Generic[I, O], Consumer[I], Producer[O], ABC):
    """Interface for components that both consume and produce data."""
