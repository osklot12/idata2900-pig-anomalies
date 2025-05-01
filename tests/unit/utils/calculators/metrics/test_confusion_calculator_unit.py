import pytest

from src.utils.calculators.metrics.confusion_calculator import ConfusionCalculator


@pytest.fixture
def predictions():
    """Fixture to provide a list of predictions."""
    return [
        0, 0, 4, 0, 3, 2, 3, 1, 0, 4, 2, 1
    ]

@pytest.fixture
def targets():
    """Fixture to provide a list of targets."""
    return [
        0, 1, 2, 0, 3, 2, 2, 1, 0, 4, 1, 4
    ]

@pytest.mark.unit
def test_compute_matrix_successfully(predictions, targets):
    """Tests that a confusion matrix is computed correctly."""
    # act
    matrix = ConfusionCalculator.calculate(predictions, targets, num_classes=4)

    # assert
    assert matrix[0][0] == 3

    assert matrix[1][0] == 1
    assert matrix[1][1] == 1
    assert matrix[1][2] == 1

    assert matrix[2][2] == 1
    assert matrix[2][3] == 1
    assert matrix[2][4] == 1

    assert matrix[3][3] == 1

    assert matrix[4][1] == 1
    assert matrix[4][4] == 1

    print(f"\n{matrix}")