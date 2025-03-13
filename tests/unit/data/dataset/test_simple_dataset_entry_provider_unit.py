import pytest

from src.data.dataset.simple_dataset_entry_provider import SimpleDatasetEntryProvider
from tests.utils.dummies.dummy_dataset_source import DummyDatasetSource


@pytest.fixture
def dataset_source():
    """Fixture for creating a dummy DatasetSource instance."""
    return DummyDatasetSource()

def test_initialization(dataset_source):
    """Tests that initialization does not raise an exception."""
    SimpleDatasetEntryProvider(dataset_source, ["mp4"], ["json"])