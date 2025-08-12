import torch
from torch import nn
from typing import Dict, List, Union, Tuple

from src.model_loader.model_structure import ModelStructureDetector

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
    detector = ModelStructureDetector(model)
    
    for i in layer_idxs:
        layer = detector.get_layer(i)
        if layer is None:
            print(f"Warning: Could not find layer {i} for residual patching")
            continue
            
        key = f"residual_L{i}"
        if key not in patch_values:
            print(f"Warning: No patch value found for {key}")
            continue
            
        val = patch_values[key]
        def _patch(module, inp, out, val=val):
            return _patch_output(out, val)
        handles.append(layer.register_forward_hook(_patch))
    return handles

def register_mlp_patch_hook(
    model: nn.Module,
    layer_idxs: List[int],
    patch_values: Dict[str, torch.Tensor]
) -> List[nn.modules.module.Module]:
    handles = []
    detector = ModelStructureDetector(model)
    
    for i in layer_idxs:
        mlp = detector.get_mlp(i)
        if mlp is None:
            print(f"Warning: Could not find MLP in layer {i} for patching")
            continue
            
        key = f"mlp_L{i}"
        if key not in patch_values:
            print(f"Warning: No patch value found for {key}")
            continue
            
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
    detector = ModelStructureDetector(model)
    
    for i in layer_idxs:
        attn = detector.get_attention(i)
        if attn is None:
            print(f"Warning: Could not find attention in layer {i} for patching")
            continue
            
        key = f"attn_L{i}"
        if key not in patch_values:
            print(f"Warning: No patch value found for {key}")
            continue
            
        val = patch_values[key]
        def _patch(module, inp, out, val=val):
            return _patch_output(out, val)
        handles.append(attn.register_forward_hook(_patch))
    return handles

def register_logits_patch_hook(
    model: nn.Module,
    patch_values: Dict[str, torch.Tensor]
) -> nn.modules.module.Module:
    detector = ModelStructureDetector(model)
    head = detector.get_lm_head()
    
    if head is None:
        raise ValueError("Could not find language model head for patching")
        
    if "logits" not in patch_values:
        raise ValueError("No patch value found for logits")
        
    val = patch_values["logits"]
    def _patch(module, inp, out, val=val):
        return val.to(out.device)
    return head.register_forward_hook(_patch)
