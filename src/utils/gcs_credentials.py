from dataclasses import dataclass

@dataclass(frozen=True)
class GCSCredentials:
    """
    Google Cloud Storage (GCS) bucket credentials.

    Attributes:
        bucket_name (str): the name of the bucket
        service_account_path (str): the path of the service account json file
    """
    bucket_name: str
    service_account_path: str