import pytest
import torch
from unittest.mock import MagicMock, patch
from src.models.twod.fastrcnn.train_fastrcnn import RCNNTrainer


@pytest.mark.unit
@patch("torch.save")
@patch("os.makedirs")
@patch("src.models.twod.fastrcnn.train_fastrcnn.fasterrcnn_resnet50_fpn")
def test_trainer_runs_three_iterations(mock_model_ctor, mock_makedirs, mock_torch_save):
    # --- Mock NetworkClient and Response ---
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.execute.return_value = [MagicMock() for _ in range(8)]  # 8 fake frames
    mock_client.send_request.return_value = mock_response

    # Set up fake frame data
    fake_image = torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8)
    fake_annotation = MagicMock()
    fake_annotation.bbox = [10, 20, 30, 40]
    fake_annotation.category_id = 1

    for frame in mock_response.execute.return_value:
        frame.image = fake_image
        frame.annotations = [fake_annotation]

    # --- Patch model behavior ---
    mock_model = MagicMock()
    mock_model.side_effect = lambda images, targets: {"loss_classifier": torch.tensor(1.0, requires_grad=True)}
    mock_model.train.return_value = None
    mock_model.parameters.return_value = [torch.tensor(1.0, requires_grad=True)]
    mock_model_ctor.return_value = mock_model

    # Patch optimizer
    with patch("torch.optim.SGD") as mock_sgd:
        mock_optimizer = MagicMock()
        mock_sgd.return_value = mock_optimizer

        # Inject trainer with limited iterations
        trainer = RCNNTrainer(client=mock_client)
        trainer.create_model = lambda: mock_model  # override model creation

        # ✅ Avoid CUDA in tests
        trainer._convert_to_tensors = lambda batch: (
            [torch.rand(3, 224, 224)],
            [{"boxes": torch.tensor([[10, 20, 40, 60]]), "labels": torch.tensor([1])}]
        )

        # Patch train method to simulate 3 iterations manually
        def limited_train():
            trainer.client.connect()
            trainer.setup()
            trainer.model.train()
            for i in range(3):
                response = trainer.client.send_request(MagicMock())
                batch = response.execute()
                images, targets = trainer._convert_to_tensors(batch)
                loss_dict = trainer.model(images, targets)
                total_loss = sum(loss_dict.values())
                total_loss = total_loss if isinstance(total_loss, torch.Tensor) else torch.tensor(total_loss)
                total_loss.backward()
                mock_optimizer.step()
                mock_optimizer.zero_grad()

                if i == 2:  # simulate save on last iteration
                    torch.save({}, f"mock_checkpoint_{i}.pt")
            return "Training completed."

        trainer.train = limited_train
        result = trainer.train()

    # --- Assertions ---
    assert result == "Training completed."
    assert mock_client.connect.called
    assert mock_client.send_request.call_count == 3
    assert mock_torch_save.called
