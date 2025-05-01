from abc import ABC, abstractmethod
from typing import TypeVar, Generic

# type to read from
T = TypeVar("T")

class JSONConverter(Generic[T], ABC):
    """Reads json files."""

    @abstractmethod
    def get_json(self, data: T) -> dict:
        """
        Returns a JSON file.

        Args:
            data (T): data to convert to json

        Returns:
            dict: dictionary for json file
        """
        raise NotImplementedError