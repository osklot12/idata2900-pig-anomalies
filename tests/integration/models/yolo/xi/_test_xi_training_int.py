import pytest
import os
import shutil
import tempfile
import torch
from torch.utils.data import IterableDataset

from src.models.twod.yolo.xi.yoloxi_training_setup import TrainingSetup


class DummyDataset(IterableDataset):
    def __iter__(self):
        img = torch.zeros(3, 640, 640)
        img[:, 240:400, 240:400] = 1.0  # white square in center

        bbox = torch.tensor([[0.5, 0.5, 0.25, 0.25, 0.0]])  # centered box
        for _ in range(10):
            yield {
                "img": img.clone(),
                "instances": {
                    "cls": torch.tensor([0]),
                    "bboxes": bbox.clone(),
                },
                "batch_idx": torch.tensor([0]),
                "im_file": ["dummy.jpg"],
                "ori_shape": [torch.tensor([640, 640])],
                "ratio_pad": [
                    (
                        torch.tensor([1.0, 1.0]),
                        torch.tensor([0.0, 0.0])
                    )
                ],
            }

    def __len__(self):
        return 10


@pytest.mark.integration
def test_yolov11_trains_and_logs_to_tensorboard():
    tmp_log_dir = tempfile.mkdtemp(prefix="tensorboard_test_")

    train_dataset = DummyDataset()
    eval_dataset = DummyDataset()  # use separate but identical dummy set

    setup = TrainingSetup(dataset=train_dataset, eval_dataset=eval_dataset, epochs=4, log_dir=tmp_log_dir)
    setup.train()

    metrics = setup.metrics
    print("📊 Final Evaluation Metrics:", metrics)

    shutil.rmtree(tmp_log_dir)
