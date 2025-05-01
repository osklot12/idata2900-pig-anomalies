import numpy as np
import pytest
from src.utils.calculators.metrics.iou_calculator import compute_iou_matrix


@pytest.mark.unit
def test_iou_matrix_basic_overlap():
    pred = np.array([[0, 0, 2, 2]])
    gt = np.array([[1, 1, 3, 3]])
    expected_iou = 1 / 7
    iou = compute_iou_matrix(pred, gt)
    np.testing.assert_allclose(iou, [[expected_iou]], rtol=1e-6)

@pytest.mark.unit
def test_iou_matrix_perfect_match():
    pred = np.array([[0, 0, 1, 1]])
    gt = np.array([[0, 0, 1, 1]])
    iou = compute_iou_matrix(pred, gt)
    np.testing.assert_allclose(iou, [[1.0]])

@pytest.mark.unit
def test_iou_matrix_no_overlap():
    pred = np.array([[0, 0, 1, 1]])
    gt = np.array([[2, 2, 3, 3]])
    iou = compute_iou_matrix(pred, gt)
    np.testing.assert_allclose(iou, [[0.0]])

@pytest.mark.unit
def test_iou_matrix_multiple_boxes():
    pred = np.array([[0, 0, 2, 2], [1, 1, 3, 3]])
    gt = np.array([[1, 1, 3, 3]])
    expected = [
        [1 / 7],
        [1.0]
    ]
    iou = compute_iou_matrix(pred, gt)
    np.testing.assert_allclose(iou, expected, rtol=1e-6)

@pytest.mark.unit
def test_iou_matrix_empty_inputs():
    empty = np.empty((0, 4))
    gt = np.array([[0, 0, 1, 1]])
    pred = np.array([[0, 0, 1, 1]])
    assert compute_iou_matrix(empty, gt).shape == (0, 1)
    assert compute_iou_matrix(pred, empty).shape == (1, 0)
    assert compute_iou_matrix(empty, empty).shape == (0, 0)
