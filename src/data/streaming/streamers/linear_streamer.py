from abc import ABC
from typing import TypeVar, Generic

from src.data.pipeline.producer import Producer
from src.data.streaming.streamers.streamer import Streamer

T = TypeVar("T")


class LinearStreamer(Generic[T], Streamer, Producer[T], ABC):
    """An interface for streamers that have one consumer, creating a linear flow of data."""