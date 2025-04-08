from typing import TypeVar, Optional, Generic

from src.data.dataset.streams.stream import Stream
from src.data.preprocessing.augmentation.augmentor import Augmentor
from src.data.streaming.feedables.feedable import Feedable
from src.data.structures.rab_pool import RABPool

T = TypeVar("T")


class PoolStream(Generic[T], Stream[T], Feedable[T]):
    """Stream reading randomly from a pool of instances."""

    def __init__(self, pool_size: int = 3000, augmentor: Optional[Augmentor[T]] = None):
        """
        Initializes a PoolStream instance.


        Args:
            pool_size (int): the max number of instances in the pool at a time, 3000 by default
            augmentor (Optional[Augmentor[T]]): the augmentor to use, None by default
        """
        self._pool = RABPool[T](maxsize=pool_size, min_ready=pool_size)
        self._augmentor = augmentor

    def feed(self, food: Optional[T]) -> None:
        if self._augmentor:
            for instance in self._augmentor.augment(food):
                self._pool.put(instance)

        else:
            self._pool.put(food)

    def read(self) -> Optional[T]:
        return self._pool.get()