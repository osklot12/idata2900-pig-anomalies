import numpy as np
import pytest

from src.utils.calculators.metrics.precision_calculator import PrecisionCalculator


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
        (0, 1/2),
        (1, 1),
        (2, 1/3)
    ]
)
def test_calculate_precision_successfully(confusion_matrix, cls, expected):
    """Tests that precision is calculated correctly for each class."""
    # act
    precision = PrecisionCalculator.calculate(confusion_matrix, cls)

    # assert
    assert precision == expected