import pytest

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
def gcs_loader_factory(bucket_name):
    """Fixture to provide a GCSLoaderFactory instance."""
    return GCSLoaderFactory(bucket_name, DummyAuthServiceFactory())


def test_create_video_loader(gcs_loader_factory, bucket_name):
    """Tests that create_video_loader() correctly instantiates GCSVideoLoader."""
    # act
    video_loader = gcs_loader_factory.create_video_loader()

    # assert
    assert isinstance(video_loader, GCSVideoLoader)
    assert video_loader._bucket_name == bucket_name
    assert isinstance(video_loader._auth_service, DummyAuthService)


def test_create_annotation_loader(gcs_loader_factory, bucket_name):
    """Tests that create_annotation_loader() correctly instantiates GCSAnnotationLoader."""
    # act
    annotation_loader = gcs_loader_factory.create_annotation_loader()

    # assert
    assert isinstance(annotation_loader, GCSAnnotationLoader)
    assert annotation_loader._bucket_name == bucket_name
    assert isinstance(annotation_loader._auth_service, DummyAuthService)

def test_create_dataset_source(gcs_loader_factory, bucket_name):
    """Tests that create_dataset_source() correctly instantiates GCSFileManager"""