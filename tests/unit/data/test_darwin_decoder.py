import pytest
import json
from src.data.darwin_decoder import DarwinDecoder

@pytest.fixture
def sample_json():
    """Provides a sample Darwin JSON annotation."""
    json_data = """
    {
        "annotations": [
            {
                "name": "g2b_bellynosing",
                "frames": {
                    "56": {
                        "bounding_box": {
                            "x": 1925.0824,
                            "y": 1178.3059,
                            "w": 108.3765,
                            "h": 110.6824
                        },
                        "keyframe": true
                    },
                    "110": {
                        "bounding_box": {
                            "x": 1925.0824,
                            "y": 1178.3059,
                            "w": 108.3765,
                            "h": 110.6824
                        },
                        "keyframe": true
                    },
                    "128": {
                        "bounding_box": {
                            "x": 1920.4706,
                            "y": 1194.4471,
                            "w": 126.8235,
                            "h": 112.9882
                        },
                        "keyframe": true
                    },
                    "162": {
                        "bounding_box": {
                            "x": 1920.4706,
                            "y": 1194.4471,
                            "w": 126.8235,
                            "h": 112.9882
                        },
                        "keyframe": true
                    }
                }
            }
        ]
    }
    """
    return json.loads(json_data)

def test_decode(sample_json):
    """Tests DarwinDecoder with a sample JSON structure."""
    decoded_annotations = DarwinDecoder.get_annotations(sample_json)

    expected_output = {
        56: [("g2b_bellynosing", 1925.0824, 1178.3059, 108.3765, 110.6824)],
        110: [("g2b_bellynosing", 1925.0824, 1178.3059, 108.3765, 110.6824)],
        128: [("g2b_bellynosing", 1920.4706, 1194.4471, 126.8235, 112.9882)],
        162: [("g2b_bellynosing", 1920.4706, 1194.4471, 126.8235, 112.9882)],
    }

    assert decoded_annotations == expected_output, "Decoded annotations do not match expected output"