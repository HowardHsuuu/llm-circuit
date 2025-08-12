import argparse
import os
import torch
import json
import sys
from typing import Dict

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.analysis.attribution import compute_importances
from src.analysis.prune import prune_top_k, prune_threshold

def main():
    parser = argparse.ArgumentParser(
        description="Compute importances and prune patching results"
    )
    parser.add_argument(
        "--patch_path",
        type=str,
        required=True,
        help="Path to patching results (.pt file)"
    )
    parser.add_argument(
        "--out_importances",
        type=str,
        default="outputs/analysis/importances.pt",
        help="Where to save importance scores"
    )
    parser.add_argument(
        "--out_pruned",
        type=str,
        default="outputs/analysis/pruned.json",
        help="Where to save pruned results"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--top_k",
        type=int,
        help="Number of top components to keep"
    )
    group.add_argument(
        "--threshold",
        type=float,
        help="Threshold for keeping components"
    )
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out_importances), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_pruned), exist_ok=True)
    
    print(f"[1/4] Loading patching deltas from {args.patch_path}")
    try:
        patch_results: Dict[str, float] = torch.load(args.patch_path, map_location='cpu')
        print(f"Loaded {len(patch_results)} patching results")
    except Exception as e:
        print(f"Error loading patching results: {e}")
        return
    
    print("[2/4] Computing absolute importances")
    try:
        importances = compute_importances(patch_results)
        print(f"Computed importances for {len(importances)} components")
    except Exception as e:
        print(f"Error computing importances: {e}")
        return
    
    if args.top_k is not None:
        print(f"[3/4] Pruning to top {args.top_k} components")
        try:
            pruned = prune_top_k(importances, args.top_k)
            print(f"Pruned to {len(pruned)} components")
        except Exception as e:
            print(f"Error pruning to top k: {e}")
            return
    else:
        print(f"[3/4] Pruning with threshold >= {args.threshold}")
        try:
            pruned = prune_threshold(importances, args.threshold)
            print(f"Pruned to {len(pruned)} components")
        except Exception as e:
            print(f"Error pruning with threshold: {e}")
            return
    
    print(f"[4/4] Saving importances to {args.out_importances}")
    try:
        torch.save(importances, args.out_importances)
        print("Importances saved successfully!")
    except Exception as e:
        print(f"Error saving importances: {e}")
        return

    print(f"[4/4] Saving pruned components to {args.out_pruned}")
    try:
        with open(args.out_pruned, "w", encoding="utf-8") as f:
            json.dump(pruned, f, indent=2, ensure_ascii=False)
        print("Pruned results saved successfully!")
    except Exception as e:
        print(f"Error saving pruned results: {e}")
        return

    print("Done.")

if __name__ == "__main__":
    main()
