from unittest.mock import Mock

import pytest

from src.data.dataclasses.dataset_instance import DatasetInstance
from src.data.dataset.providers.manifest_instance_provider import ManifestInstanceProvider


def _make_incrementing_string_generator(prefix: str = "id"):
    """Creates an incrementing string generator."""
    i = 0
    while True:
        yield f"{prefix}_{i}"
        i += 1


@pytest.fixture
def selector():
    """Fixture to provide a Manifest instance."""
    gen = _make_incrementing_string_generator()
    mock = Mock()
    mock.next.side_effect = lambda: next(gen)
    return mock


@pytest.fixture
def manifest():
    """Fixture to provide a Manifest instance."""
    video_gen = _make_incrementing_string_generator("video")
    annotation_gen = _make_incrementing_string_generator("annotation")

    mock = Mock()
    mock.get_instance.side_effect = lambda _: DatasetInstance(
        video_file=next(video_gen),
        annotation_file=next(annotation_gen)
    )
    return mock


@pytest.fixture
def manifest_instance_provider(manifest, selector):
    """Fixture to provide a ManifestInstanceProvider instance."""
    return ManifestInstanceProvider(
        manifest=manifest,
        selector=selector
    )


@pytest.mark.unit
def test_next_returns_the_next_instance(manifest_instance_provider):
    """Tests that calling next returns the next expected instance."""
    # act
    instances = [manifest_instance_provider.next() for _ in range(10)]

    # assert
    for i in range(len(instances)):
        instance = instances[i]
        assert isinstance(instance, DatasetInstance)
        assert instance.video_file == f"video_{i}"
        assert instance.annotation_file == f"annotation_{i}"


@pytest.mark.unit
def test_next_returns_none_when_selector_returns_none(manifest):
    """Tests that calling next returns None when the internal selector returns None."""
    # arrange
    selector = Mock()
    selector.next.return_value = None

    provider = ManifestInstanceProvider(
        manifest=manifest,
        selector=selector
    )

    # act
    instance = provider.next()

    # assert
    assert instance is None


@pytest.mark.unit
def test_next_raises_when_manifest_returns_none(selector):
    """Tests that calling next raises when the internal manifest returns None."""
    # arrange
    manifest = Mock()
    manifest.get_instance.return_value = None

    provider = ManifestInstanceProvider(
        manifest=manifest,
        selector=selector
    )

    # act & assert
    with pytest.raises(RuntimeError):
        provider.next()
