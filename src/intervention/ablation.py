import torch
from torch import nn
from typing import List

def register_residual_ablation_hook(
    model: nn.Module,
    layer_idxs: List[int]
) -> List[nn.modules.module.Module]:
    handles = []
    for i in layer_idxs:
        block = model.transformer.h[i]

        def _ablate(module, inp, out):
            return torch.zeros_like(out)

        handle = block.register_forward_hook(_ablate)
        handles.append(handle)
    return handles


def register_mlp_ablation_hook(
    model: nn.Module,
    layer_idxs: List[int]
) -> List[nn.modules.module.Module]:
    handles = []
    for i in layer_idxs:
        mlp = model.transformer.h[i].mlp

        def _ablate(module, inp, out):
            return torch.zeros_like(out)

        handle = mlp.register_forward_hook(_ablate)
        handles.append(handle)
    return handles


def register_attn_ablation_hook(
    model: nn.Module,
    layer_idxs: List[int]
) -> List[nn.modules.module.Module]:
    handles = []
    for i in layer_idxs:
        attn = model.transformer.h[i].self_attn

        def _ablate(module, inp, out):
            return torch.zeros_like(out)

        handle = attn.register_forward_hook(_ablate)
        handles.append(handle)
    return handles
