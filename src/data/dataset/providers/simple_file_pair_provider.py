from typing import Optional

from src.data.dataclasses.file_pair import FilePair
from src.data.dataset.matching.matching_strategy import MatchingStrategy
from src.data.dataset.providers.file_pair_provider import FilePairProvider
from src.data.dataset.registries.file_registry import FileRegistry


class SimpleFilePairProvider(FilePairProvider):
    """Simple file pair provider."""

    def __init__(self, file_registry: FileRegistry, video_suffixes: List[str], annotation_suffixes matcher: MatchingStrategy):

    def get_next(self) -> Optional[FilePair]:
        pass

