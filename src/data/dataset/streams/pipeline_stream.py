from typing import TypeVar, Generic, Optional, Iterable

from src.data.dataset.streams.closable import Closable
from src.data.dataset.streams.closable_stream import ClosableStream
from src.data.dataset.streams.stream import Stream
from src.data.pipeline.pipeline_builder import PipelineBuilder
from src.data.pipeline.sink import Sink

# source output data type
A = TypeVar("A")

# pipeline output data type
B = TypeVar("B")


class PipelineStream(Generic[A, B], ClosableStream[B]):
    """Stream adapter for pipelines, allowing for pulling data through pipelines."""

    def __init__(self, source: ClosableStream[A], pipeline: PipelineBuilder[A, B]):
        """
        Initializes a PipelineStream instance.

        Args:
            source (Stream[A]): stream to pull data from
            pipeline (PipelineBuilder[A, B]): pipeline to push data through
        """
        self._source = source
        self._sink = Sink()
        self._pipeline = pipeline.into(self._sink)

    def read(self) -> Optional[B]:
        if self._sink.is_empty():
            data = self._source.read()
            self._pipeline.consume(data)

        return self._sink.get()

    def close(self) -> None:
        self._source.close()