import os
from typing import List, Optional

from src.data.dataset.matching.matching_strategy import MatchingStrategy
from src.data.parsing.base_name_parser import BaseNameParser


class BaseNameMatcher(MatchingStrategy):
    """Matches files based on their base name, having a valid suffix."""

    def __init__(self):
        """Initializes a BaseNameMatcher instance."""
        self._parser = BaseNameParser()

    def match(self, reference: str, candidates: List[str]) -> Optional[str]:
        if reference is None:
            raise ValueError("file_name cannot be None")

        if candidates is None:
            raise ValueError("candidates cannot be None")

        ref_base = self._parser.parse_string(reference)

        return next((c for c in candidates if self._parser.parse_string(c) == ref_base), None)