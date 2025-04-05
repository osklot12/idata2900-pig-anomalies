from abc import ABC, abstractmethod
from typing import List, Optional


class FileSelector(ABC):
    """Strategy for selecting a file from a dataset."""

    @abstractmethod
    def select_file(self) -> Optional[str]:
        """
        Selects a file from the dataset.

        Returns:
            Optional[str]: the selected file, or None if no valid file is found
        """
        raise NotImplementedError