import torch
from torch import nn
from typing import Dict, List, Union, Tuple

TensorTuple = Union[torch.Tensor, Tuple]

def _patch_output(out: TensorTuple, val: torch.Tensor) -> TensorTuple:
    if isinstance(out, tuple):
        device = out[0].device
        patched = val.to(device)
        return (patched, *out[1:])
    else:
        return val.to(out.device)

def register_residual_patch_hook(
    model: nn.Module,
    layer_idxs: List[int],
    patch_values: Dict[str, torch.Tensor]
) -> List[nn.modules.module.Module]:
    handles = []
    for i in layer_idxs:
        block = model.model.layers[i]
        key = f"residual_L{i}"
        val = patch_values[key]
        def _patch(module, inp, out, val=val):
            return _patch_output(out, val)
        handles.append(block.register_forward_hook(_patch))
    return handles

def register_mlp_patch_hook(
    model: nn.Module,
    layer_idxs: List[int],
    patch_values: Dict[str, torch.Tensor]
) -> List[nn.modules.module.Module]:
    handles = []
    for i in layer_idxs:
        mlp = model.model.layers[i].mlp
        key = f"mlp_L{i}"
        val = patch_values[key]
        def _patch(module, inp, out, val=val):
            return _patch_output(out, val)
        handles.append(mlp.register_forward_hook(_patch))
    return handles

def register_attn_patch_hook(
    model: nn.Module,
    layer_idxs: List[int],
    patch_values: Dict[str, torch.Tensor]
) -> List[nn.modules.module.Module]:
    handles = []
    for i in layer_idxs:
        attn = model.model.layers[i].self_attn
        key = f"attn_L{i}"
        val = patch_values[key]
        def _patch(module, inp, out, val=val):
            return _patch_output(out, val)
        handles.append(attn.register_forward_hook(_patch))
    return handles

def register_logits_patch_hook(
    model: nn.Module,
    patch_values: Dict[str, torch.Tensor]
) -> nn.modules.module.Module:
    head = model.lm_head if hasattr(model, "lm_head") else model.model.lm_head
    val = patch_values["logits"]
    def _patch(module, inp, out, val=val):
        return val.to(out.device)
    return head.register_forward_hook(_patch)
