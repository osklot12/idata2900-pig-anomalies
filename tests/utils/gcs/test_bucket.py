from src.utils.path_finder import PathFinder


class TestBucket:
    """Real Google Cloud Storage bucket for integration testing."""
    BUCKET_NAME = "norsvin-g2b-behavior-prediction"

    SERVICE_ACCOUNT_FILE = PathFinder.get_abs_path(".secrets/service-account.json")

    SAMPLE_VIDEO = "g2b_behaviour/images/avd13_cam1_20220314072829_20220314073013_fps2.0.mp4"

    SAMPLE_ANNOTATION = "g2b_behaviour/releases/g2b-prediction/annotations/avd13_cam1_20220314072829_20220314073013_fps2.0.json"