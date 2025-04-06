from abc import ABC, abstractmethod
from typing import List, Optional


class MatchingStrategy(ABC):
    """Interface for different file matching strategies."""

    @abstractmethod
    def match(self, reference: str, candidates: List[str]) -> Optional[str]:
        """
        Finds a matching file for the given file_name from a list of candidates.

        Args:
            reference (str): the reference to find a match for
            candidates (List[str]): the list potential matches

        Returns:
            Optional[str]: the matching file, or None if no match is found
        """
        raise NotImplementedError