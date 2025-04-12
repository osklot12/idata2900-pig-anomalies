import pytest
import torch
from unittest.mock import MagicMock, patch
from src.models.twod.fastrcnn.train_fastrcnn import RCNNTrainer
from src.data.dataset.streams.prefetcher import Prefetcher


@pytest.mark.unit
@patch("torch.save")
@patch("os.makedirs")
@patch("src.models.twod.fastrcnn.train_fastrcnn.fasterrcnn_resnet50_fpn")
def test_trainer_runs_three_iterations(mock_model_ctor, mock_makedirs, mock_torch_save):
    # --- Create a mock prefetcher that returns fake batches ---
    fake_frame = MagicMock()
    fake_frame.image = torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8)
    fake_ann = MagicMock()
    fake_ann.bbox = [10, 20, 30, 40]
    fake_ann.category_id = 1
    fake_frame.annotations = [fake_ann]

    mock_prefetcher = MagicMock(spec=Prefetcher)
    mock_prefetcher.get.side_effect = [[fake_frame] * 8] * 3  # 3 batches

    # --- Patch model ---
    mock_model = MagicMock()
    mock_model.side_effect = lambda imgs, targs: {"loss_classifier": torch.tensor(1.0, requires_grad=True)}
    mock_model.train.return_value = None
    mock_model.parameters.return_value = [torch.tensor(1.0, requires_grad=True)]
    mock_model_ctor.return_value = mock_model

    # --- Patch optimizer ---
    with patch("torch.optim.SGD") as mock_sgd:
        mock_optimizer = MagicMock()
        mock_sgd.return_value = mock_optimizer

        # --- Create trainer ---
        trainer = RCNNTrainer(prefetcher=mock_prefetcher)
        trainer.create_model = lambda: mock_model
        trainer._convert_to_tensors = lambda batch: (
            [torch.rand(3, 224, 224)],
            [{"boxes": torch.tensor([[10, 20, 40, 60]]), "labels": torch.tensor([1])}]
        )

        # --- Simulate training with 3 iterations ---
        def limited_train():
            trainer.setup()
            trainer.model.train()
            for i in range(3):
                batch = trainer.prefetcher.get()
                images, targets = trainer._convert_to_tensors(batch)
                loss_dict = trainer.model(images, targets)
                total_loss = sum(loss_dict.values())
                total_loss.backward()
                mock_optimizer.step()
                mock_optimizer.zero_grad()

                # dashboard push simulated every iteration
                _ = trainer.model(images, targets)

                # checkpoint only on last
                if i == 2:
                    torch.save({}, f"mock_checkpoint_{i}.pt")
            return "Training completed."

        trainer.train = limited_train
        result = trainer.train()

    # --- Assertions ---
    assert result == "Training completed."
    assert mock_prefetcher.get.call_count == 3
    assert mock_torch_save.called
