import pytest

from src.auth.gcp_auth_service import GCPAuthService
from src.data.dataset.manifests.matching_manifest import MatchingManifest
from src.data.dataset.registries.gcs_file_registry import GCSFileRegistry
from src.data.dataset.registries.suffix_file_registry import SuffixFileRegistry
from src.data.parsing.base_name_parser import BaseNameParser
from tests.utils.gcs.test_bucket import TestBucket


@pytest.fixture
def auth_service():
    """Fixture to provide an AuthService instance."""
    return GCPAuthService(TestBucket.SERVICE_ACCOUNT_FILE)


@pytest.fixture
def gcs_file_registry(auth_service):
    """Fixture to provide a GCSFileRegistry instance."""
    return GCSFileRegistry(TestBucket.BUCKET_NAME, auth_service)


@pytest.fixture
def video_registry(gcs_file_registry):
    """Fixture to provide a video file registry."""
    return SuffixFileRegistry(
        source=gcs_file_registry,
        suffixes=("mp4",)
    )


@pytest.fixture
def annotations_registry(gcs_file_registry):
    """Fixture to provide an annotations file registry."""
    return SuffixFileRegistry(
        source=gcs_file_registry,
        suffixes=("json",)
    )


@pytest.fixture
def manifest(video_registry, annotations_registry):
    """Fixture to provide a MatchingManifest instance."""
    return MatchingManifest(
        video_registry=video_registry,
        annotations_registry=annotations_registry
    )


@pytest.mark.integration
def test_getting_all_instances(manifest):
    """Tests that all retrieved instances are paired as expected."""
    # arrange
    parser = BaseNameParser()

    # act
    ids = manifest.ids
    instances = [manifest.get_instance(id_) for id_ in ids]

    # assert
    for instance in instances:
        assert parser.parse_string(instance.video_file) == parser.parse_string(instance.annotation_file)