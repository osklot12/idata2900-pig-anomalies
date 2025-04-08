from typing import TypeVar, Optional, Generic

from src.data.dataset.streams.stream import Stream
from src.data.preprocessing.augmentation.augmentor import Augmentor
from src.data.streaming.feedables.feedable import Feedable

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
        self._pool_size = pool_size
        self._augmentor = augmentor


    def feed(self, food: Optional[T]) -> None:


    def read(self) -> Optional[T]:
        pass