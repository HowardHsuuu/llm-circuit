import argparse
import os
import torch
from typing import Dict, List

from circuit_tracing_llama.model_loader.llama_loader import LlamaModelWrapper
from circuit_tracing_llama.intervention.patching import (
    register_residual_patch_hook,
    register_mlp_patch_hook,
    register_attn_patch_hook,
)
from circuit_tracing_llama.intervention.patching import register_logits_patch_hook

def main():
    parser = argparse.ArgumentParser(
        description="Run activation patching to compute logit deltas"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help=""
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=""
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help=""
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help=""
    )
    parser.add_argument(
        "--acts_path",
        type=str,
        required=True,
        help=""
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="outputs/patching/patch_results.pt",
        help=""
    )
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    print(f"[1/5] Loading activations from {args.acts_path}")
    patch_values: Dict[str, torch.Tensor] = torch.load(args.acts_path)
    print(f"[2/5] Loading model {args.model} on {args.device}...")
    loader = LlamaModelWrapper(args.model, device=args.device)
    model = loader.model
    tokenizer = loader.tokenizer
    print(f"[3/5] Computing original logits for prompt: \"{args.prompt}\"")
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = model(**inputs)
    original_logits = out.logits[0, -1, :]
    print("[4/5] Running patching experiments...")
    target_pos = -1
    components = [k for k in patch_values.keys() if k.startswith(("residual_L","mlp_L","attn_L"))]

    patch_results: Dict[str, float] = {}

    for comp in components:
        layer = int(comp.split("_L")[1])
        handles = []
        if comp.startswith("residual_L"):
            handles = register_residual_patch_hook(model, [layer], patch_values)
        elif comp.startswith("mlp_L"):
            handles = register_mlp_patch_hook(model, [layer], patch_values)
        elif comp.startswith("attn_L"):
            handles = register_attn_patch_hook(model, [layer], patch_values)

        with torch.no_grad():
            out_p = model(**inputs)
        patched_logits = out_p.logits[0, -1, :]
        idx = torch.argmax(original_logits).item()
        delta = (patched_logits[idx] - original_logits[idx]).item()
        patch_results[comp] = delta
        for h in handles:
            h.remove()

    print(f"[5/5] Saving patching deltas ({len(patch_results)} items) to {args.out_path}")
    torch.save(patch_results, args.out_path)
    print("Done.")

if __name__ == "__main__":
    main()
