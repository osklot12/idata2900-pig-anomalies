import pytest
import torch
from unittest.mock import MagicMock, patch, call
from src.models.twod.fastrcnn.train_fastrcnn import RCNNTrainer


@pytest.mark.unit
@patch("torch.save")
@patch("os.makedirs")
def test_trainer_runs_full_training_loop(mock_makedirs, mock_torch_save):
    # --- Mock dependencies ---
    mock_client = MagicMock()
    mock_context = MagicMock()

    mock_response = MagicMock()
    mock_response.execute.return_value = [MagicMock() for _ in range(8)]  # Mocked AnnotatedFrames

    mock_client.send_request.return_value = mock_response

    # Fake image and annotation setup
    fake_image = torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8)
    fake_annotation = MagicMock()
    fake_annotation.bbox = [10, 20, 30, 40]
    fake_annotation.category_id = 1

    for frame in mock_response.execute.return_value:
        frame.image = fake_image
        frame.annotations = [fake_annotation]

    # Patch model and its forward call
    with patch("src.models.train_fastrcnn.fasterrcnn_resnet50_fpn") as mock_model_ctor:
        mock_model = MagicMock()
        mock_model.return_value = {"loss_classifier": torch.tensor(1.0, requires_grad=True)}
        mock_model.__call__.return_value = {"loss_classifier": torch.tensor(1.0, requires_grad=True)}
        mock_model_ctor.return_value = mock_model
        mock_model.train.return_value = None
        mock_model.parameters.return_value = [torch.tensor(1.0, requires_grad=True)]

        # Patch optimizer
        with patch("torch.optim.SGD") as mock_sgd:
            mock_optimizer = MagicMock()
            mock_sgd.return_value = mock_optimizer

            # --- Run trainer ---
            trainer = RCNNTrainer(client=mock_client, context=mock_context)
            trainer.create_model = lambda: mock_model  # override model creation
            result = trainer.train()

    # --- Assertions ---
    assert result == "Training completed."
    assert mock_client.connect.called
    assert mock_client.send_request.call_count == 1000
    assert mock_torch_save.call_args_list[-1][0][1].endswith("model_final.pt")
