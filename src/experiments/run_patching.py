import argparse
import os
import torch
import sys
from typing import Dict, List

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.model_loader.flexible_loader import FlexibleModelLoader
from src.intervention.patching import (
    register_residual_patch_hook,
    register_mlp_patch_hook,
    register_attn_patch_hook,
    register_logits_patch_hook,
)

def main():
    parser = argparse.ArgumentParser(
        description="Run activation patching to compute logit deltas"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/DialoGPT-medium",  # Smaller, more accessible model
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on: cpu, cuda, or mps"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt for patching experiment"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Which layers to patch (default: all available)"
    )
    parser.add_argument(
        "--acts_path",
        type=str,
        required=True,
        help="Path to saved activations (.pt file)"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="outputs/patching/patch_results.pt",
        help="Where to save patching results"
    )
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    
    print(f"[1/5] Loading activations from {args.acts_path}")
    try:
        patch_values: Dict[str, torch.Tensor] = torch.load(args.acts_path, map_location='cpu')
        print(f"Loaded {len(patch_values)} activation tensors")
    except Exception as e:
        print(f"Error loading activations: {e}")
        return
    
    print(f"[2/5] Loading model {args.model} on {args.device}...")
    try:
        loader = FlexibleModelLoader(args.model, device=args.device)
        model = loader.model
        tokenizer = loader.tokenizer
        
        # Print model information
        model_info = loader.get_model_info()
        print(f"Model loaded successfully:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print(f"[3/5] Computing original logits for prompt: \"{args.prompt}\"")
    try:
        inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = model(**inputs)
        original_logits = out.logits[0, -1, :]
        print(f"Original logits shape: {original_logits.shape}")
    except Exception as e:
        print(f"Error computing original logits: {e}")
        return
    
    print("[4/5] Running patching experiments...")
    
    # Determine which layers to patch
    if args.layers is None:
        # Try to infer from activations
        layer_indices = set()
        for key in patch_values.keys():
            if key.startswith(("residual_L", "mlp_L", "attn_L")):
                try:
                    layer_idx = int(key.split("_L")[1])
                    layer_indices.add(layer_idx)
                except (ValueError, IndexError):
                    continue
        args.layers = sorted(list(layer_indices))
        print(f"Inferred layers to patch from activations: {args.layers}")
    
    if not args.layers:
        print("Warning: No layers found to patch")
        return
    
    # Get components to patch
    components = [k for k in patch_values.keys() if k.startswith(("residual_L","mlp_L","attn_L"))]
    print(f"Found {len(components)} components to patch")
    
    patch_results: Dict[str, float] = {}
    
    for comp in components:
        try:
            # Extract layer index from component name
            if "_L" not in comp:
                print(f"Warning: Skipping component {comp} - no layer info")
                continue
                
            layer = int(comp.split("_L")[1])
            if layer not in args.layers:
                continue
                
            handles = []
            
            if comp.startswith("residual_L"):
                handles = register_residual_patch_hook(model, [layer], patch_values)
            elif comp.startswith("mlp_L"):
                handles = register_mlp_patch_hook(model, [layer], patch_values)
            elif comp.startswith("attn_L"):
                handles = register_attn_patch_hook(model, [layer], patch_values)
            
            if not handles:
                print(f"Warning: No hooks registered for {comp}")
                continue
            
            # Run patched forward pass
            with torch.no_grad():
                out_p = model(**inputs)
            patched_logits = out_p.logits[0, -1, :]
            
            # Compute delta for the most likely token
            idx = torch.argmax(original_logits).item()
            delta = (patched_logits[idx] - original_logits[idx]).item()
            patch_results[comp] = delta
            
            # Clean up hooks
            for h in handles:
                h.remove()
                
        except Exception as e:
            print(f"Error patching component {comp}: {e}")
            continue
    
    print(f"[5/5] Saving patching deltas ({len(patch_results)} items) to {args.out_path}")
    try:
        torch.save(patch_results, args.out_path)
        print(f"Patching results saved successfully!")
        
        # Print summary of results
        if patch_results:
            print("Patching results summary:")
            for comp, delta in sorted(patch_results.items(), key=lambda x: abs(x[1]), reverse=True):
                print(f"  {comp}: {delta:.6f}")
        else:
            print("No successful patching results")
            
    except Exception as e:
        print(f"Error saving patching results: {e}")
    
    print("Done.")

if __name__ == "__main__":
    main()
