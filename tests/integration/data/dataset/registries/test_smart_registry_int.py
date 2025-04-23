from typing import Dict

import pytest

from src.auth.factories.gcp_auth_service_factory import GCPAuthServiceFactory
from src.data.dataset.label.factories.simple_label_parser_factory import SimpleLabelParserFactory
from src.data.dataset.registries.smart_registry import SmartRegistry
from src.data.decoders.factories.darwin_decoder_factory import DarwinDecoderFactory
from src.data.loading.loaders.factories.gcs_loader_factory import GCSLoaderFactory
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from src.utils.norsvin_dataset_config import NORSVIN_GCS_CREDS

@pytest.fixture
def loader_factory():
    """Fixture to provide a LoaderFactory instance."""
    gcs_creds = NORSVIN_GCS_CREDS
    auth_factory = GCPAuthServiceFactory(service_account_file=gcs_creds.service_account_path)
    decoder_factory = DarwinDecoderFactory(
        label_parser_factory=SimpleLabelParserFactory(NorsvinBehaviorClass.get_label_map())
    )

    return GCSLoaderFactory(
        bucket_name=gcs_creds.bucket_name,
        auth_factory=auth_factory,
        decoder_factory=decoder_factory
    )

@pytest.fixture
def filter_func():
    """Fixture to provide a filter function."""
    def has_annotations(data: Dict[NorsvinBehaviorClass, int]) -> bool:
        return any(count > 0 for count in data.values())

    return has_annotations


def test_get_file_paths_success(loader_factory, filter_func):
    """Tests that file paths are successfully filtered as desired."""
    # arrange
    registry = SmartRegistry(
        loader_factory=loader_factory,
        filter_func=filter_func
    )

    files = registry.get_file_paths()
    for file in files:
        print(file)