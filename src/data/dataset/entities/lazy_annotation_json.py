from src.data.dataset.entities.annotation_json import AnnotationJson
from src.data.loading.loaders.annotation_loader import AnnotationLoader


class LazyAnnotationJson(AnnotationJson):
    """An annotation that lazy loads the annotation data."""

    def __init__(self, annotation_id: str, annotation_loader: AnnotationLoader):
        """
        Initializes a LazyAnnotation instance.

        Args:
            annotation_id (str): the annotation ID
            annotation_loader (AnnotationLoader): the annotation loader
        """
        self._id = annotation_id
        self._loader = annotation_loader

    def get_id(self) -> str:
        return self._id

    def get_data(self) -> dict:
        return self._loader.load_annotation(self._id)