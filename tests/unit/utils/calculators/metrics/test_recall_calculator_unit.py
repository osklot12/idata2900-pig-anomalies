import numpy as np
import pytest

from src.utils.calculators.metrics.recall_calculator import RecallCalculator


@pytest.fixture
def confusion_matrix():
    """Fixture to provide a confusion matrix."""
    return np.array([
        [2, 0, 1],
        [1, 2, 1],
        [1, 0, 1]
    ])

@pytest.mark.unit
@pytest.mark.parametrize(
    "cls, expected",
    [
        (0, 2/3),
        (1, 1/2),
        (2, 1/2)
    ]
)
def test_calculate_recall_successfully(confusion_matrix, cls, expected):
    """Tests that recall is calculated correctly for each class."""
    # act
    recall = RecallCalculator.calculate(confusion_matrix, cls)

    # assert
    assert recall == expected