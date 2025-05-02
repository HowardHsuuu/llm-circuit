import torch
from torch import nn
from typing import Dict, Union

TensorOrTuple = Union[torch.Tensor, tuple]

def _unwrap_output(out: TensorOrTuple) -> torch.Tensor:
    if isinstance(out, tuple):
        return out[0]
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    return out

def register_residual_hook(
    model: nn.Module,
    layer_idxs: list[int],
    activations: Dict[str, torch.Tensor],
    prefix: str = "residual"
) -> list[nn.modules.module.Module]:
    handles = []
    for i in layer_idxs:
        block = model.model.layers[i]
        name = f"{prefix}_L{i}"
        def _hook(module, inp, out, key=name):
            tensor = _unwrap_output(out)
            activations[key] = tensor.detach().cpu()
        handles.append(block.register_forward_hook(_hook))
    return handles

def register_mlp_hook(
    model: nn.Module,
    layer_idxs: list[int],
    activations: Dict[str, torch.Tensor],
    prefix: str = "mlp"
) -> list[nn.modules.module.Module]:
    handles = []
    for i in layer_idxs:
        mlp = model.model.layers[i].mlp
        name = f"{prefix}_L{i}"
        def _hook(module, inp, out, key=name):
            tensor = _unwrap_output(out)
            activations[key] = tensor.detach().cpu()
        handles.append(mlp.register_forward_hook(_hook))
    return handles

def register_attn_hook(
    model: nn.Module,
    layer_idxs: list[int],
    activations: Dict[str, torch.Tensor],
    prefix: str = "attn"
) -> list[nn.modules.module.Module]:
    handles = []
    for i in layer_idxs:
        attn = model.model.layers[i].self_attn
        name = f"{prefix}_L{i}"
        def _hook(module, inp, out, key=name):
            tensor = _unwrap_output(out)
            activations[key] = tensor.detach().cpu()
        handles.append(attn.register_forward_hook(_hook))
    return handles

def register_logits_hook(
    model: nn.Module,
    activations: Dict[str, torch.Tensor],
    prefix: str = "logits"
) -> nn.modules.module.Module:
    head = model.lm_head if hasattr(model, "lm_head") else model.model.lm_head
    name = prefix
    def _hook(module, inp, out):
        tensor = _unwrap_output(out)
        activations[name] = tensor.detach().cpu()
    return head.register_forward_hook(_hook)
