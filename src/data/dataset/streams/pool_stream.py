from typing import TypeVar, Optional, Generic, List

from src.data.dataset.streams.stream import Stream
from src.data.preprocessing.preprocessor import Preprocessor
from src.data.pipeline.consumer import Consumer
from src.data.structures.rab_pool import RABPool

T = TypeVar("T")


class PoolStream(Generic[T], Stream[T], Consumer[T]):
    """Stream reading randomly from a pool of instances."""

    def __init__(self, pool_size: int = 3000, preprocessors: Optional[List[Preprocessor[T]]] = None):
        """
        Initializes a PoolStream instance.


        Args:
            pool_size (int): the max number of instances in the pool at a time, 3000 by default
            preprocessors (Optional[List[Preprocessor[T]]]): optional preprocessors to use, None by default
        """
        self._pool = RABPool[T](maxsize=pool_size, min_ready=pool_size)
        self._preprocessors = preprocessors if preprocessors is not None else []

    def consume(self, data: Optional[T]) -> None:
        instances = [data]
        for preprocessor in self._preprocessors:
            new_instances = []
            for instance in instances:
                processed = preprocessor.process(instance)
                new_instances.extend(processed)
            instances = new_instances

        for instance in instances:
            self._pool.put(instance)

    def read(self) -> Optional[T]:
        return self._pool.get()