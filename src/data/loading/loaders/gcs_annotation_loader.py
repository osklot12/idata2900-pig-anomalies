from src.data.loading.loaders.annotation_loader import AnnotationLoader
from src.data.gcs_bucket_client import GCSBucketClient


class GCSAnnotationLoader(GCSBucketClient, AnnotationLoader):
    """Handles downloading and parsing annotation files from Google Cloud Storage."""

    def load_annotation(self, annotation_id: str) -> dict:
        return self._make_request(
            self._get_file_url(annotation_id)
        ).json()