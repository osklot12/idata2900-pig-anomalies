import numpy as np
import pytest

from src.utils.calculators.metrics.f1_calculator import F1Calculator


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
        (0, 4/7),
        (1, 2/3),
        (2, 2/5)
    ]
)
def test_calculate_f1_successfully(confusion_matrix, cls, expected):
    """Tests that recall is calculated correctly for each class."""
    # act
    f1 = F1Calculator.calculate(confusion_matrix, cls)

    # assert
    assert f1 == pytest.approx(expected, rel=1e-6)