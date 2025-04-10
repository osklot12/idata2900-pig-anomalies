from typing import List

from src.data.dataset.selectors.factories.string_selector_factory import StringSelectorFactory
from src.data.dataset.selectors.random_string_selector import RandomStringSelector
from src.data.dataset.selectors.string_selector import StringSelector


class RandomStringSelectorFactory(StringSelectorFactory):
    """Factory for creating RandomStringSelector instances."""

    def __init__(self, strings: List[str]):
        """
        Initializes a RandomStringSelectorFactory instance.

        Args:
            strings (List[str]): list of strings to select from
        """
        self._strings = strings

    def create_selector(self) -> StringSelector:
        return RandomStringSelector(list(self._strings))
