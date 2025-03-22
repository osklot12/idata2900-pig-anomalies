import os
from typing import List, Optional

from src.data.dataset.matching.matching_strategy import MatchingStrategy


class BaseNameMatchingStrategy(MatchingStrategy):
    """Matches files based on their base name, having a valid suffix."""

    def __init__(self, suffixes: List[str]):
        """
        Initializes a BaseNameMatching instance.

        Args:
            suffixes (List[str]): a list of valid suffixes for the matching file
        """
        self._suffixes = suffixes

    def find_match(self, file_name: str, candidates: List[str]) -> Optional[str]:
        if file_name is None:
            raise ValueError("file_name cannot be None")

        if candidates is None:
            raise ValueError("candidates cannot be None")

        base_name = os.path.splitext(os.path.basename(file_name))[0]

        match = next(
            (
                match_path for match_path in candidates
                if os.path.splitext(os.path.basename(match_path))[0] == base_name
                   and any(match_path.endswith(suffix) for suffix in self._suffixes)
            ), None
        )

        print(f"[BaseNameMatchingStrategy] Matched file {match}")

        return match