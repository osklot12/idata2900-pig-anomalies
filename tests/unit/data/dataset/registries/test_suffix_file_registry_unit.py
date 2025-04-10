from typing import List

import pytest

from src.data.dataset.registries.file_registry import FileRegistry
from src.data.dataset.registries.suffix_file_registry import SuffixFileRegistry


class DummyFileRegistry(FileRegistry):
    """Dummy FileRegistry implementation."""

    def __init__(self, files: List[str]):
        self.files = files

    def get_file_paths(self) -> set[str]:
        return set(self.files)


@pytest.fixture
def files():
    """Fixture to provide test files."""
    return [
        "dira/dirb/file1.mp4", "dira/dirb/file2.mp4", "dira/dirb/file3.txt",
        "dirb/dirc/file4.mp4", "dirb/dirc/file5.mp3", "dirb/dirc/file6.mp4"
    ]


@pytest.fixture
def registry(files):
    """Fixture to provide a DummyFileRegistry instance."""
    return DummyFileRegistry(files=files)


@pytest.mark.unit
def test_filters_out_by_suffixes(registry, files):
    """Tests that SuffixFileRegistry filters out files by suffixes."""
    # arrange
    suffix_registry = SuffixFileRegistry(source=registry, suffixes=("mp4",))

    # act
    result = suffix_registry.get_file_paths()

    # assert
    assert len(result) == 4
    assert files[0] in result
    assert files[1] in result
    assert files[2] not in result
    assert files[3] in result
    assert files[4] not in result
    assert files[5] in result
