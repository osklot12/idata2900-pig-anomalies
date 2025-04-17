from typing import List

from src.data.dataset.selectors.determ_string_selector import DetermStringSelector
from src.data.dataset.selectors.factories.selector_factory import SelectorFactory
from src.data.dataset.selectors.selector import Selector


class DetermStringSelectorFactory(SelectorFactory[str]):
    """Factory for creating DetermStringSelector instances."""

    def create_selector(self, candidates: List[str]) -> Selector[str]:
        return DetermStringSelector(strings=candidates)