from unittest.mock import Mock

import pytest

from src.data.dataset.sources.gcs_dataset_source import GCSDatasetSource
from src.data.loading.factories.gcs_loader_factory import GCSLoaderFactory
from src.data.loading.loaders.gcs_annotation_loader import GCSAnnotationLoader
from src.data.loading.loaders.gcs_video_loader import GCSVideoLoader
from tests.utils.dummies.dummy_auth_service import DummyAuthService
from tests.utils.dummies.dummy_auth_service_factory import DummyAuthServiceFactory


@pytest.fixture
def bucket_name():
    """Fixture to provide a test bucket name."""
    return "test-bucket"


@pytest.fixture
def decoder_factory():
    """Fixture to provide a mock DecoderFactory instance."""
    return Mock()


@pytest.fixture
def gcs_loader_factory(bucket_name, decoder_factory):
    """Fixture to provide a GCSLoaderFactory instance."""
    return GCSLoaderFactory(bucket_name, DummyAuthServiceFactory(), decoder_factory)


@pytest.mark.unit
def test_create_video_loader(gcs_loader_factory, bucket_name):
    """Tests that create_video_loader() correctly instantiates GCSVideoLoader."""
    # act
    video_loader = gcs_loader_factory.create_video_loader()

    # assert
    assert isinstance(video_loader, GCSVideoLoader)
    assert video_loader.get_bucket_name() == bucket_name
    assert isinstance(video_loader.get_auth_service(), DummyAuthService)


@pytest.mark.unit
def test_create_annotation_loader(gcs_loader_factory, bucket_name):
    """Tests that create_annotation_loader() correctly instantiates GCSAnnotationLoader."""
    # act
    annotation_loader = gcs_loader_factory.create_annotation_loader()

    # assert
    assert isinstance(annotation_loader, GCSAnnotationLoader)
    assert annotation_loader.get_bucket_name() == bucket_name
    assert isinstance(annotation_loader.get_auth_service(), DummyAuthService)


@pytest.mark.unit
def test_create_dataset_source(gcs_loader_factory, bucket_name):
    """Tests that create_dataset_source() correctly instantiates GCSDatasetSource."""
    # act
    dataset_source = gcs_loader_factory.create_dataset_source()

    # assert
    assert isinstance(dataset_source, GCSDatasetSource)
    assert dataset_source.get_bucket_name() == bucket_name
    assert isinstance(dataset_source.get_auth_service(), DummyAuthService)
