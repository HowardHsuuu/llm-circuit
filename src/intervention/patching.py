import torch
from torch import nn
from typing import Dict, List, Any

def register_residual_patch_hook(
    model: nn.Module,
    layer_idxs: List[int],
    patch_values: Dict[str, torch.Tensor]
) -> List[nn.modules.module.Module]:
    handles = []
    for i in layer_idxs:
        block = model.transformer.h[i]
        key = f"residual_L{i}"
        value = patch_values[key]

        def _patch(module, inp, out, val=value):
            return val.to(out.device)

        handle = block.register_forward_hook(_patch)
        handles.append(handle)
    return handles


def register_mlp_patch_hook(
    model: nn.Module,
    layer_idxs: List[int],
    patch_values: Dict[str, torch.Tensor]
) -> List[nn.modules.module.Module]:
    handles = []
    for i in layer_idxs:
        mlp = model.transformer.h[i].mlp
        key = f"mlp_L{i}"
        val = patch_values[key]

        def _patch(module, inp, out, val=val):
            return val.to(out.device)

        handle = mlp.register_forward_hook(_patch)
        handles.append(handle)
    return handles


def register_attn_patch_hook(
    model: nn.Module,
    layer_idxs: List[int],
    patch_values: Dict[str, torch.Tensor]
) -> List[nn.modules.module.Module]:
    handles = []
    for i in layer_idxs:
        attn = model.transformer.h[i].self_attn
        key = f"attn_L{i}"
        val = patch_values[key]

        def _patch(module, inp, out, val=val):
            return val.to(out.device)

        handle = attn.register_forward_hook(_patch)
        handles.append(handle)
    return handles


def register_logits_patch_hook(
    model: nn.Module,
    patch_values: Dict[str, torch.Tensor]
) -> nn.modules.module.Module:
    head = model.lm_head if hasattr(model, "lm_head") else model.model.lm_head
    val = patch_values["logits"]

    def _patch(module, inp, out, val=val):
        return val.to(out.device)

    handle = head.register_forward_hook(_patch)
    return handle
