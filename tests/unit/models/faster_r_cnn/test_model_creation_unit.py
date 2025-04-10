import torch
import pytest
from torchvision.models.detection import fasterrcnn_resnet50_fpn

@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_creation_on_cuda():
    device = torch.device("cuda")
    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=2).to(device)

    for param in model.parameters():
        assert param.device.type == "cuda"
