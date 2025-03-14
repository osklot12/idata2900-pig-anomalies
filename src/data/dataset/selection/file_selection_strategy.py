from abc import ABC, abstractmethod
from typing import List, Optional


class FileSelectionStrategy(ABC):
    """Strategy for selecting a file from a dataset."""

    @abstractmethod
    def select_file(self, candidates: List[str]) -> Optional[str]:
        """
        Selects a file from the list of candidates.

        Args:
            candidates (List[str]): list of available files

        Returns:
            Optional[str]: the selected file, or None if no valid file is found
        """
        raise NotImplementedError