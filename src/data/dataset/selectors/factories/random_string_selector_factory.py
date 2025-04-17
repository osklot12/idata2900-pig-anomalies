from typing import List

from src.data.dataset.selectors.factories.selector_factory import SelectorFactory, T
from src.data.dataset.selectors.random_string_selector import RandomStringSelector
from src.data.dataset.selectors.selector import Selector


class RandomStringSelectorFactory(SelectorFactory[str]):
    """Factory for creating RandomStringSelector instances."""

    def create_selector(self, candidates: List[str]) -> Selector[str]:
        return RandomStringSelector(strings=candidates)