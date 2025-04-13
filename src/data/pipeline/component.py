from abc import ABC
from typing import TypeVar

from src.data.pipeline.consumer import Consumer
from src.data.pipeline.producer import Producer

T = TypeVar("T")

class Component(Consumer[T], Producer[T], ABC):
    """Interface for pipeline components, consuming and producing the same type of data."""