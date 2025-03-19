from unittest.mock import Mock

import pytest

from src.auth.gcp_auth_service import GCPAuthService
from src.data.preprocessing.normalization.simple_bbox_normalizer import SimpleBBoxNormalizer
from src.data.decoders.darwin_decoder import DarwinDecoder
from tests.utils.gcs.test_bucket import TestBucket


@pytest.fixture
def gcp_data_loader():
    """Fixture to provide an actual GCPDataLoader."""
    auth_service = GCPAuthService(credentials_path=TestBucket.SERVICE_ACCOUNT_FILE)
    return GCSDataLoaderOld(bucket_name=NorsvinBucketParser.BUCKET_NAME, auth_service=auth_service)

@pytest.fixture
def mock_callback():
    """Fixture to provide a mock callback."""
    return Mock()

@pytest.mark.integration
def test_real_gcp_annotations_processing(gcp_data_loader, mock_callback):
    """Integration test: verify real GCP annotation processing."""
    # arrange
    normalizer = SimpleBBoxNormalizer(image_dimensions=(2688, 1502), new_range=(0, 1), annotation_parser=NorsvinLabelParser)
    annotation_loader = AnnotationStreamer(
        data_loader=gcp_data_loader,
        annotation_blob_name=NorsvinBucketParser.get_annotation_blob_name(SampleBucketFiles.SAMPLE_JSON_FILE),
        decoder_cls=DarwinDecoder,
        normalizer=normalizer,
        callback=mock_callback
    )

    # act
    annotation_loader.start_streaming()
    annotation_loader.wait_for_completion()

    # assert
    assert mock_callback.call_count > 1, "Callback should be called at least once!"

    last_call = mock_callback.call_args_list[-1]
    _, last_frame_index, last_annotation, is_complete = last_call[0]

    assert last_annotation is None, "Final callback should have annotation=None!"
    assert is_complete is True, "Final callback should indicate completion!"