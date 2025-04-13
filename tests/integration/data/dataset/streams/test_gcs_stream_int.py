import pytest

from src.data.dataclasses.dataset_split_ratios import DatasetSplitRatios
from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.factories.gcs_stream_factory import GCSStreamFactory
from src.utils.gcs_credentials import GCSCredentials
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from tests.utils.gcs.test_bucket import TestBucket


def test_norsvin_train_stream():
    """Tests that creating the Norsvin training set stream with GCSStreamFactory gives a working stream."""
    # arrange
    gcs_creds = GCSCredentials(bucket_name=TestBucket.BUCKET_NAME, service_account_path=TestBucket.SERVICE_ACCOUNT_FILE)
    split_ratios = DatasetSplitRatios(train=0.8, val=0.2, test=0.2)



    stream = GCSStreamFactory(
        gcs_creds=gcs_creds,
        split_ratios=split_ratios,
        split=DatasetSplit.TRAIN,
        label_map=NorsvinBehaviorClass.get_label_map(),
    )