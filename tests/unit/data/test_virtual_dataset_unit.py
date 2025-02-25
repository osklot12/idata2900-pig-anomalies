import pytest

from src.data.virtual_dataset import VirtualDataset
from tests.utils.dummies.dummy_gcp_data_loader import DummyGCPDataLoader


@pytest.fixture
def dummy_data_loader():
    """Creates a DummyGCPDataLoader instance for testing."""
    return DummyGCPDataLoader(bucket_name="test-bucket", credentials_path="test.json")

@pytest.fixture
def virtual_dataset(dummy_data_loader):
    """Creates a VirtualDataset instance for testing."""
    return VirtualDataset(
        loader=dummy_data_loader,
        max_sources=10,
        max_frames_per_source=20
    )