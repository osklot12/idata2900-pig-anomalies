import pytest
import os
import shutil
import tempfile
import torch
from torch.utils.data import IterableDataset, DataLoader

from src.models.twod.yolo.xi.yoloxi_training_setup import TrainingSetup


class DummyDataset(IterableDataset):
    def __iter__(self):
        for i in range(10):
            yield {
                "img": torch.rand(3, 640, 640),
                "instances": {
                    "cls": torch.tensor([0]),
                    "bboxes": torch.rand(1, 5),
                },
                "batch_idx": torch.tensor([0]),
                "im_file": [f"dummy_{i}.jpg"],
                "ori_shape": [torch.tensor([640, 640])],  # ✅ tensor inside a list
                "ratio_pad": [  # ✅ list with one tuple of tensors
                    (
                        torch.tensor([1.0, 1.0]),  # gain (width, height)
                        torch.tensor([0.0, 0.0])   # pad (width, height)
                    )
                ],
            }

    def __len__(self):
        return 10


@pytest.mark.integration
def test_yolov11_trains_and_logs_to_tensorboard():
    """Trains real YOLOv11 for 1 epoch with dummy data and verifies tensorboard logging."""
    tmp_log_dir = tempfile.mkdtemp(prefix="tensorboard_test_")
    dataset = DummyDataset()

    # Use real TrainingSetup logic (no patching of train)
    setup = TrainingSetup(dataset=dataset, epochs=1, log_dir=tmp_log_dir)
    setup.train()

    # Verify TensorBoard log output
    log_files = os.listdir(setup.log_dir)
    assert any(f.startswith("events.out.tfevents") for f in log_files), \
        f"No tensorboard logs found in {setup.log_dir}"

    # Cleanup
    shutil.rmtree(tmp_log_dir)
