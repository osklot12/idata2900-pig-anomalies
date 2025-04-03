import pytest
from unittest.mock import MagicMock, patch, call

from src.models.twod.yolo.xi.yoloxi_training_setup import TrainingSetup


@pytest.mark.unit
@patch("src.models.twod.yolo.xi.yoloxi_training_setup.SummaryWriter")
@patch("src.models.twod.yolo.xi.yoloxi_training_setup.YOLO")
def test_log_epoch_metrics_logs_scalars(YOLOMock, SummaryWriterMock):
    """Tests that _log_epoch_metrics logs per-epoch scalars to TensorBoard."""

    # arrange
    fake_writer = MagicMock()
    SummaryWriterMock.return_value = fake_writer

    # Create a TrainingSetup instance (training won't actually run)
    setup = TrainingSetup(dataset=MagicMock(), epochs=1)

    # Simulate a trainer object with epoch and metrics
    trainer_mock = MagicMock()
    trainer_mock.epoch = 5
    trainer_mock.metrics = {
        "loss/cls": 0.2,
        "loss/box": 0.4,
        "mAP50": 0.85,
    }

    # act
    setup._log_epoch_metrics(trainer_mock)

    # assert
    fake_writer.add_scalar.assert_has_calls([
        call("metrics/loss/cls", 0.2, 5),
        call("metrics/loss/box", 0.4, 5),
        call("metrics/mAP50", 0.85, 5),
    ], any_order=True)
