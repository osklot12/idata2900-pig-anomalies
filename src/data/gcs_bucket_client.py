import requests
from requests.exceptions import HTTPError

from src.auth.auth_service import AuthService


class GCSBucketClient:
    """A base class for loaders loading data from Google Cloud Storage (GCS)."""

    def __init__(self, bucket_name: str, auth_service: AuthService):
        """
        Initializes a GCSDataLoader instance.

        Args:
            bucket_name (str): the name of the bucket
            auth_service (AuthService): the authentication service
        """
        self._bucket_name = bucket_name
        self._auth_service = auth_service

    def _get_headers(self) -> dict:
        """Generates authentication headers for GCS requests."""
        return {"Authorization": f"Bearer {self._auth_service.get_access_token()}"}

    def _get_file_url(self, blob_name: str) -> str:
        """Constructs the full GCS URL for a given file."""
        return f"https://storage.googleapis.com/{self._bucket_name}/{blob_name}"

    def _make_request(self, url: str, stream: bool = False) -> requests.Response:
        """
        Makes a GET request to fetch a file from GCS.

        Args:
            url (str): the endpoint to make a GET request for
            stream (bool): whether to enable streaming (for large files)

        Returns:
            requests.Response: the HTTP response
        """
        headers = self._get_headers()
        url = url

        response = requests.get(url, headers=headers, stream=stream)

        try:
            response.raise_for_status()
        except HTTPError as e:
            raise HTTPError(f"Request to '{url}' failed: {e}")

        return response