from unittest.mock import Mock

import pytest

from src.auth.gcp_auth_service import GCPAuthService
from src.data.bbox_normalizer import BBoxNormalizer
from src.data.decoders.darwin_decoder import DarwinDecoder
from src.data.streaming.streamers import AnnotationStreamer
from src.data.loading.gcp_data_loader import GCPDataLoader
from src.utils.norsvin_annotation_parser import NorsvinAnnotationParser
from src.utils.norsvin_bucket_parser import NorsvinBucketParser
from tests.utils.constants.sample_bucket_files import SampleBucketFiles


@pytest.fixture
def gcp_data_loader():
    """Fixture to provide an actual GCPDataLoader."""
    auth_service = GCPAuthService(credentials_path=NorsvinBucketParser.CREDENTIALS_PATH)
    return GCPDataLoader(bucket_name=NorsvinBucketParser.BUCKET_NAME, auth_service=auth_service)

@pytest.fixture
def mock_callback():
    """Fixture to provide a mock callback."""
    return Mock()

@pytest.mark.integration
def test_real_gcp_annotations_processing(gcp_data_loader, mock_callback):
    """Integration test: verify real GCP annotation processing."""
    # arrange
    normalizer = BBoxNormalizer(image_dimensions=(2688, 1502), new_range=(0, 1), annotation_parser=NorsvinAnnotationParser)
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