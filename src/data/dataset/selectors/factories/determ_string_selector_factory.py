from typing import List

from src.data.dataset.selectors.determ_string_selector import DetermStringSelector
from src.data.dataset.selectors.factories.string_selector_factory import StringSelectorFactory
from src.data.dataset.selectors.string_selector import StringSelector


class DetermStringSelectorFactory(StringSelectorFactory):
    """Factory for creating DetermStringSelector instances."""

    def __init__(self, strings: List[str]):
        """
        Initializes a DetermStringSelectorFactory instance.

        Args:
            strings (List[str]): list of strings to select from
        """
        self._strings = strings

    def create_selector(self) -> StringSelector:
        return DetermStringSelector(list(self._strings))