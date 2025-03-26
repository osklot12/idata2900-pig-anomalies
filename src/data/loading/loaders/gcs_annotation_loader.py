from typing import List

from src.auth.auth_service import AuthService
from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.decoders.annotation_decoder import AnnotationDecoder
from src.data.gcs_bucket_client import GCSBucketClient
from src.data.loading.loaders.video_annotations_loader import VideoAnnotationsLoader


class GCSAnnotationLoader(GCSBucketClient, VideoAnnotationsLoader):
    """Handles downloading and parsing annotation files from Google Cloud Storage."""

    def __init__(self, bucket_name: str, auth_service: AuthService, decoder: AnnotationDecoder):
        """
        Initializes a GCSAnnotationLoader instance.

        Args:
            bucket_name (str): the name of the bucket
            auth_service (AuthService): the authentication service
            decoder (AnnotationDecoder): the annotation decoder
        """
        super().__init__(bucket_name, auth_service)
        self._decoder = decoder

    def load_video_annotations(self, annotations_id: str) -> List[FrameAnnotations]:
        raw_data = self._make_request(self._get_file_url(annotations_id)).content
        print(f"[GCSAnnotationLoader] Loaded annotation file: {annotations_id}")
        return self._decoder.decode_annotations(raw_data)