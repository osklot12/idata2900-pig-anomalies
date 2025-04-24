import types
import torch
import torch.nn as nn

def safe_concat_forward(self, x):
    if isinstance(x, torch.Tensor):
        return x  # already a single tensor
    return torch.cat(x, self.d)  # expected case: list/tuple of tensors

def patch_concat_modules(model: nn.Module):
    """
    Monkey-patches all Concat layers in the given model to prevent torch.cat errors.
    """
    for module in model.modules():
        if module.__class__.__name__ == "Concat":
            module.forward = types.MethodType(safe_concat_forward, module)
