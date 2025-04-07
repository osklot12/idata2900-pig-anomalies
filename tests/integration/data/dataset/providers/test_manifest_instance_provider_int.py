import pytest

from src.auth.gcp_auth_service import GCPAuthService
from src.data.dataset.manifests.matching_manifest import MatchingManifest
from src.data.dataset.providers.manifest_instance_provider import ManifestInstanceProvider
from src.data.dataset.registries.gcs_file_registry import GCSFileRegistry
from src.data.dataset.registries.suffix_file_registry import SuffixFileRegistry
from src.data.dataset.selectors.determ_string_selector import DetermStringSelector
from src.data.dataset.splitters.determ_splitter import DetermSplitter
from tests.utils.gcs.test_bucket import TestBucket


@pytest.fixture
def auth_service():
    """Fixture to provide an AuthService instance."""
    return GCPAuthService(TestBucket.SERVICE_ACCOUNT_FILE)


@pytest.fixture
def gcs_file_registry(auth_service):
    """Fixture to provide a GCSFileRegistry instance."""
    return GCSFileRegistry(bucket_name=TestBucket.BUCKET_NAME, auth_service=auth_service)


@pytest.fixture
def video_registry(gcs_file_registry):
    """Fixture to provide a video registry."""
    return SuffixFileRegistry(source=gcs_file_registry, suffixes=("mp4",))


@pytest.fixture
def annotations_registry(gcs_file_registry):
    """Fixture to provide an annotations registry."""
    return SuffixFileRegistry(source=gcs_file_registry, suffixes=("json",))


@pytest.fixture
def manifest(video_registry, annotations_registry):
    """Fixture to provide a Manifest instance."""
    return MatchingManifest(video_registry=video_registry, annotations_registry=annotations_registry)


@pytest.fixture
def splitter(manifest):
    """Fixture to provide a DetermSplitter instance."""
    return DetermSplitter(manifest.ids, [0.8, 0.1, 0.1])


@pytest.mark.integration
def test_split_providers_are_consistent(manifest, splitter):
    """Tests that the ManifestInstanceProvider for each split is consistent with a deterministic splitter."""
    # arrange
    splits_list = [[], [], []]

    # act
    for i in range(len(splits_list)):
        for _ in range(10):
            provider = ManifestInstanceProvider(manifest, DetermStringSelector(splitter.splits[i]))

            instances = []
            instance = provider.next()
            while instance is not None:
                instances.append(instance)
                instance = provider.next()

            splits_list[i].append(instances)

    # assert
    for split_runs in splits_list:
        first_run = split_runs[0]
        for run in split_runs[1:]:
            assert run == first_run