import torch
from torch import nn
from typing import Dict, Union, List

from src.model_loader.model_structure import ModelStructureDetector

TensorOrTuple = Union[torch.Tensor, tuple]

def _unwrap_output(out: TensorOrTuple) -> torch.Tensor:
    if isinstance(out, tuple):
        return out[0]
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    return out

def register_residual_hook(
    model: nn.Module,
    layer_idxs: List[int],
    activations: Dict[str, torch.Tensor],
    prefix: str = "residual"
) -> List[nn.modules.module.Module]:
    handles = []
    detector = ModelStructureDetector(model)
    
    for i in layer_idxs:
        layer = detector.get_layer(i)
        if layer is None:
            print(f"Warning: Could not find layer {i}")
            continue
            
        name = f"{prefix}_L{i}"
        def _hook(module, inp, out, key=name):
            tensor = _unwrap_output(out)
            activations[key] = tensor.detach().cpu()
        handles.append(layer.register_forward_hook(_hook))
    return handles

def register_mlp_hook(
    model: nn.Module,
    layer_idxs: List[int],
    activations: Dict[str, torch.Tensor],
    prefix: str = "mlp"
) -> List[nn.modules.module.Module]:
    handles = []
    detector = ModelStructureDetector(model)
    
    for i in layer_idxs:
        mlp = detector.get_mlp(i)
        if mlp is None:
            print(f"Warning: Could not find MLP in layer {i}")
            continue
            
        name = f"{prefix}_L{i}"
        def _hook(module, inp, out, key=name):
            tensor = _unwrap_output(out)
            activations[key] = tensor.detach().cpu()
        handles.append(mlp.register_forward_hook(_hook))
    return handles

def register_attn_hook(
    model: nn.Module,
    layer_idxs: List[int],
    activations: Dict[str, torch.Tensor],
    prefix: str = "attn"
) -> List[nn.modules.module.Module]:
    handles = []
    detector = ModelStructureDetector(model)
    
    for i in layer_idxs:
        attn = detector.get_attention(i)
        if attn is None:
            print(f"Warning: Could not find attention in layer {i}")
            continue
            
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
    detector = ModelStructureDetector(model)
    head = detector.get_lm_head()
    
    if head is None:
        raise ValueError("Could not find language model head in the model")
        
    name = prefix
    def _hook(module, inp, out):
        tensor = _unwrap_output(out)
        activations[name] = tensor.detach().cpu()
    return head.register_forward_hook(_hook)
