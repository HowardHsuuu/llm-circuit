import torch
from torch import nn
from typing import Callable, Dict

def register_residual_hook(
    model: nn.Module,
    layer_idxs: list[int],
    activations: Dict[str, torch.Tensor],
    prefix: str = "residual"
) -> list:
    handles = []
    for i in layer_idxs:
        block = model.transformer.h[i]
        name = f"{prefix}_L{i}"
        def _hook(module, inp, out, key=name):
            activations[key] = out.detach().cpu()
        h = block.register_forward_hook(_hook)
        handles.append(h)
    return handles

def register_mlp_hook(
    model: nn.Module,
    layer_idxs: list[int],
    activations: Dict[str, torch.Tensor],
    prefix: str = "mlp"
) -> list:
    handles = []
    for i in layer_idxs:
        mlp = model.transformer.h[i].mlp
        name = f"{prefix}_L{i}"
        def _hook(module, inp, out, key=name):
            activations[key] = out.detach().cpu()
        h = mlp.register_forward_hook(_hook)
        handles.append(h)
    return handles

def register_attn_hook(
    model: nn.Module,
    layer_idxs: list[int],
    activations: Dict[str, torch.Tensor],
    prefix: str = "attn"
) -> list:
    handles = []
    for i in layer_idxs:
        attn = model.transformer.h[i].self_attn
        name = f"{prefix}_L{i}"
        def _hook(module, inp, out, key=name):
            activations[key] = out.detach().cpu()
        h = attn.register_forward_hook(_hook)
        handles.append(h)
    return handles

def register_logits_hook(
    model: nn.Module,
    activations: Dict[str, torch.Tensor],
    prefix: str = "logits"
) -> Callable:
    head = model.lm_head if hasattr(model, "lm_head") else model.model.lm_head
    name = f"{prefix}"
    def _hook(module, inp, out):
        # out shape: (batch, seq_len, vocab_size)
        activations[name] = out.detach().cpu()
    handle = head.register_forward_hook(_hook)
    return handle
