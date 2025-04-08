from typing import TypeVar, Generic, Callable, List

from src.data.preprocessing.augmentation.augmentor import Augmentor

T = TypeVar("T")


class ConditionalAugmentor(Generic[T], Augmentor[T]):
    """Augmentor that augments conditionally."""

    def __init__(self, augmentor: Augmentor[T], predicate: Callable[[T], bool]):
        """
        Initializes a ConditionalAugmentor instance.

        Args:
            augmentor (Augmentor[T]): the augmentor to use
            predicate (Callable[[T], bool]): the predicate that decides whether to augment the instance
        """
        self._augmentor = augmentor
        self._predicate = predicate

    def augment(self, instance: T) -> List[T]:
        result = [instance]

        if self._predicate(instance):
            result = self._augmentor.augment(instance)

        return result
