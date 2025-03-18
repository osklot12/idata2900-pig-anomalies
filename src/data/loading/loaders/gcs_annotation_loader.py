from src.data.dataset.entities.video_annotations import VideoAnnotations
from src.data.loading.loaders.annotation_loader import VideoAnnotationsLoader
from src.data.gcs_bucket_client import GCSBucketClient


class GCSAnnotationLoader(GCSBucketClient, VideoAnnotationsLoader):
    """Handles downloading and parsing annotation files from Google Cloud Storage."""

    def load_video_annotations(self, annotation_id: str) -> VideoAnnotations:
        return self._make_request(
            self._get_file_url(annotation_id)
        ).json()