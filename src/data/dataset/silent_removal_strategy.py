from src.data.dataset.matching_error_strategy import MatchingErrorStrategy


class SilentRemovalStrategy(MatchingErrorStrategy):
    """A matching error strategy that silently removes files on error."""

    def handle_no_match(self, file_name: str) -> None:
        self._data_provider.remove_unmatched_file(file_name)

    def handle_unknown_file(self, file_name: str) -> None:
        self._data_provider.remove_unmatched_file(file_name)
