import pytest
from unittest.mock import Mock

from src.data.dataclasses.dataset_instance import DatasetInstance
from src.data.dataset.manifests.matching_manifest import MatchingManifest


@pytest.fixture
def video_paths():
    """Fixture to provide a list of video paths."""
    return ["dir_a/dir_b/video1.mp4"]


@pytest.fixture
def annotations_paths():
    """Fixture to provide a list of annotation paths."""
    return ["dir_a/dir_b/video1.json"]


@pytest.fixture
def video_registry(video_paths):
    """Fixture to provide a video registry."""
    registry = Mock()
    registry.get_file_paths.return_value = video_paths
    return registry


@pytest.fixture
def annotations_registry(annotations_paths):
    """Fixture to provide an annotation registry."""
    registry = Mock()
    registry.get_file_paths.return_value = annotations_paths
    return registry


@pytest.fixture
def matcher(annotations_paths):
    """Fixture to provide a MatchingStrategy instance."""
    m = Mock()
    m.match.return_value = annotations_paths[0]
    return m


@pytest.fixture
def instance_id():
    """Fixture to provide an instance ID."""
    return "video1"


@pytest.fixture
def identifier(instance_id):
    """Fixture to provide an Identifier instance."""
    i = Mock()
    i.identify.return_value = instance_id
    return i


@pytest.fixture
def manifest(video_registry, annotations_registry, matcher, identifier):
    """Fixture to provide a MatchingManifest instance."""
    return MatchingManifest(
        video_registry=video_registry,
        annotations_registry=annotations_registry,
        matcher=matcher,
        identifier=identifier
    )


@pytest.mark.unit
def test_list_all_ids(manifest, instance_id):
    """Tests that list_all_ids returns the expected IDs."""
    # act
    ids = manifest.ids

    # assert
    assert len(ids) == 1
    assert instance_id in ids


@pytest.mark.unit
def test_get_instance(manifest, instance_id, video_paths, annotations_paths):
    """Tests that get_instance returns the expected instance."""
    # act
    instance = manifest.get_instance(instance_id)

    # assert
    assert isinstance(instance, DatasetInstance)
    assert instance.video_file == video_paths[0]
    assert instance.annotation_file == annotations_paths[0]
