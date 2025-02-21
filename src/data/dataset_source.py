from abc import ABC, abstractmethod
from typing import List

class DatasetSource(ABC):
    """
    A source of dataset instances.
    """

    @abstractmethod
    def list_files(self) -> List[str]:
        """
        Should return a list of available instances.
        """
        pass