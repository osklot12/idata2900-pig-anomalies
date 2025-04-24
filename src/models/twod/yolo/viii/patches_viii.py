import types
import torch
import torch.nn as nn

def strict_safe_concat_forward(self, x):
    if isinstance(x, torch.Tensor):
        x = [x]  # wrap single tensor in list to avoid torch.cat error
    elif not isinstance(x, (list, tuple)):
        raise TypeError(f"Concat input must be Tensor, list, or tuple, got {type(x)}")
    return torch.cat(tuple(x), dim=self.d)  # ✅ Use self.d

def patch_concat_modules(model: nn.Module):
    for module in model.modules():
        if module.__class__.__name__ == "Concat":
            module.forward = types.MethodType(strict_safe_concat_forward, module)
