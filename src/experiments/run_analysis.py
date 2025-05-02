import argparse
import os
import torch
import json
from typing import Dict

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
        help=""
    )
    parser.add_argument(
        "--out_importances",
        type=str,
        default="outputs/analysis/importances.pt",
        help=""
    )
    parser.add_argument(
        "--out_pruned",
        type=str,
        default="outputs/analysis/pruned.json",
        help=""
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--top_k",
        type=int,
        help=""
    )
    group.add_argument(
        "--threshold",
        type=float,
        help=""
    )
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out_importances), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_pruned), exist_ok=True)
    print(f"[1/4] Loading patching deltas from {args.patch_path}")
    patch_results: Dict[str, float] = torch.load(args.patch_path)
    print("[2/4] Computing absolute importances")
    importances = compute_importances(patch_results)
    if args.top_k is not None:
        print(f"[3/4] Pruning to top {args.top_k} components")
        pruned = prune_top_k(importances, args.top_k)
    else:
        print(f"[3/4] Pruning with threshold >= {args.threshold}")
        pruned = prune_threshold(importances, args.threshold)
    print(f"[4/4] Saving importances to {args.out_importances}")
    torch.save(importances, args.out_importances)

    print(f"[4/4] Saving pruned components to {args.out_pruned}")
    with open(args.out_pruned, "w", encoding="utf-8") as f:
        json.dump(pruned, f, indent=2, ensure_ascii=False)

    print("Done.")

if __name__ == "__main__":
    main()
