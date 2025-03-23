import pytest

from src.auth.gcp_auth_service import GCPAuthService
from src.data.dataset.providers.file_source_id_provider import FileSourceIDsProvider
from src.data.dataset.sources.gcs_dataset_source import GCSDatasetSource
from src.data.parsing.file_base_name_parser import FileBaseNameParser
from tests.utils.gcs.test_bucket import TestBucket


@pytest.fixture
def dataset_source():
    """Fixture to provide a DatasetSource instance."""
    auth_service = GCPAuthService(TestBucket.SERVICE_ACCOUNT_FILE)
    return GCSDatasetSource(
        bucket_name=TestBucket.BUCKET_NAME,
        auth_service=auth_service
    )

@pytest.fixture
def parser():
    """Fixture to provide a StringParser instance."""
    return FileBaseNameParser()

@pytest.mark.integration
def test_get_source_ids(dataset_source, parser):
    """Tests that get_source_ids() returns the parsed source IDs."""
    # arrange
    sources_before = dataset_source.list_files()
    provider = FileSourceIDsProvider(dataset_source, parser)

    # act
    sources_after = provider.get_source_ids()

    # assert
    assert len(sources_before) == len(sources_after)
    assert not sources_after == sources_before