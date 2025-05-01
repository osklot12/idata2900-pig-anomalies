import pytest
from unittest.mock import MagicMock, patch, call

from src.models.twod.yolo.xi.yoloxi_training_setup import TrainingSetup


@pytest.mark.unit
@patch("src.models.twod.yolo.xi.yoloxi_training_setup.SummaryWriter")
@patch("src.models.twod.yolo.xi.yoloxi_training_setup.OBBTrainer")
def test_training_setup_trains_and_logs(OBBTrainerMock, SummaryWriterMock):
    """Tests that TrainingSetup sets up YOLO training and logs final metrics."""

    # arrange
    fake_dataset = MagicMock()
    fake_writer = MagicMock()
    SummaryWriterMock.return_value = fake_writer

    mock_trainer = MagicMock()
    OBBTrainerMock.return_value = mock_trainer

    mock_trainer.train.return_value = None
    mock_trainer.metrics = {
        "precision": 0.9,
        "recall": 0.85,
        "mAP50": 0.88,
        "loss/cls": 0.2,
    }

    # act
    setup = TrainingSetup(dataset=fake_dataset, epochs=1)
    setup.train()

    # assert
    mock_trainer.train.assert_called_once()

    fake_writer.add_scalar.assert_has_calls([
        call("final_metrics/precision", 0.9, 1),
        call("final_metrics/recall", 0.85, 1),
        call("final_metrics/mAP50", 0.88, 1),
        call("final_metrics/loss/cls", 0.2, 1),
    ], any_order=True)

    fake_writer.flush.assert_called_once()
    fake_writer.close.assert_called_once()
