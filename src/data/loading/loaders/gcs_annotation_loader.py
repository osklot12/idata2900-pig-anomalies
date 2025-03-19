from typing import List

from src.auth.auth_service import AuthService
from src.data.dataclasses.frame_annotation import FrameAnnotation
from src.data.gcs_bucket_client import GCSBucketClient
from src.data.loading.loaders.video_annotations_loader import VideoAnnotationsLoader


class GCSAnnotationLoader(GCSBucketClient, VideoAnnotationsLoader):
    """Handles downloading and parsing annotation files from Google Cloud Storage."""

    def __init(self, bucket_name: str, auth_service: AuthService, ):

    def load_video_annotations(self, annotations_id: str) -> List[FrameAnnotation]:
        return self._make_request(
            self._get_file_url(annotations_id)
        ).json()
