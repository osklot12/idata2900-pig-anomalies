from src.data.dataclasses.dataset_split_ratios import DatasetSplitRatios
from src.utils.gcs_credentials import GCSCredentials
from src.utils.path_finder import PathFinder
from tests.utils.gcs.test_bucket import TestBucket

NORSVIN_BUCKET_NAME = "norsvin-g2b-behavior-prediction"

NORSVIN_SERVICE_ACCOUNT_FILE_PATH = str(PathFinder.get_abs_path(".secrets/service-account.json"))

NORSVIN_GCS_CREDS = GCSCredentials(bucket_name=NORSVIN_BUCKET_NAME, service_account_path=NORSVIN_SERVICE_ACCOUNT_FILE_PATH)

NORSVIN_SPLIT_RATIOS = DatasetSplitRatios(0.8, 0.1, 0.1)

NORSVIN_TRAIN_SET_SIZE = 7500

NORSVIN_VAL_SET_SIZE = 3442

NORSVIN_TEST_SET_SIZE = 6951