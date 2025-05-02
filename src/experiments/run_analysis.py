import sys, os
import argparse, torch, json
from src.analysis.attribution import compute_importances
from src.analysis.prune       import prune_top_k, prune_threshold

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patch_path",     type=str, required=True)
    p.add_argument("--out_importances",type=str, default="outputs/analysis/importances.pt")
    p.add_argument("--out_pruned",     type=str, default="outputs/analysis/pruned.json")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--top_k",     type=int)
    grp.add_argument("--threshold", type=float)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_importances), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_pruned),     exist_ok=True)

    patch_results = torch.load(args.patch_path)
    importances   = compute_importances(patch_results)
    pruned = (prune_top_k(importances, args.top_k) if args.top_k is not None else prune_threshold(importances, args.threshold))

    torch.save(importances, args.out_importances)
    with open(args.out_pruned, "w", encoding="utf-8") as f:
        json.dump(pruned, f, indent=2)

    print(f"[+] Saved importances → {args.out_importances}")
    print(f"[+] Saved pruned → {args.out_pruned}")

if __name__=="__main__":
    main()
