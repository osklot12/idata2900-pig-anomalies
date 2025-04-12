from abc import ABC
from typing import TypeVar, Generic

from src.data.pipeline.consumer import Consumer
from src.data.pipeline.producer import Producer

T = TypeVar("T")


class Preprocessor(Generic[T], Consumer[T], Producer[T], ABC):
    """Interface for preprocessors."""