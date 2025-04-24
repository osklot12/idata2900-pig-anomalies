import types
import torch
import torch.nn as nn

def strict_safe_concat_forward(self, x):
    if isinstance(x, torch.Tensor):
        x = [x]
    elif not isinstance(x, (list, tuple)):
        raise TypeError(f"Concat input must be Tensor, list, or tuple, got {type(x)}")

    shapes = [t.shape for t in x]
    print(f"🧩 Concat inputs: {[str(s) for s in shapes]}, dim={self.d}")
    return torch.cat(tuple(x), self.d)


def patch_concat_modules(model: nn.Module):
    for module in model.modules():
        if module.__class__.__name__ == "Concat":
            module.forward = types.MethodType(strict_safe_concat_forward, module)
