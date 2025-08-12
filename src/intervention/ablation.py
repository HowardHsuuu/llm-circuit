import torch
from torch import nn
from typing import List

from src.model_loader.model_structure import ModelStructureDetector

def register_residual_ablation_hook(
    model: nn.Module,
    layer_idxs: List[int]
) -> List[nn.modules.module.Module]:
    handles = []
    detector = ModelStructureDetector(model)
    
    for i in layer_idxs:
        layer = detector.get_layer(i)
        if layer is None:
            print(f"Warning: Could not find layer {i} for residual ablation")
            continue

        def _ablate(module, inp, out):
            return torch.zeros_like(out)

        handle = layer.register_forward_hook(_ablate)
        handles.append(handle)
    return handles


def register_mlp_ablation_hook(
    model: nn.Module,
    layer_idxs: List[int]
) -> List[nn.modules.module.Module]:
    handles = []
    detector = ModelStructureDetector(model)
    
    for i in layer_idxs:
        mlp = detector.get_mlp(i)
        if mlp is None:
            print(f"Warning: Could not find MLP in layer {i} for ablation")
            continue

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
    detector = ModelStructureDetector(model)
    
    for i in layer_idxs:
        attn = detector.get_attention(i)
        if attn is None:
            print(f"Warning: Could not find attention in layer {i} for ablation")
            continue

        def _ablate(module, inp, out):
            return torch.zeros_like(out)

        handle = attn.register_forward_hook(_ablate)
        handles.append(handle)
    return handles
