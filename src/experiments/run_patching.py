import argparse, os, torch
from src.model_loader.llama_loader import LlamaModelWrapper
from src.intervention.patching import Patcher

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--acts_path", type=str, required=True)
    parser.add_argument("--out_path", type=str,
                        default="outputs/patching/patch_results.pt")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    cache = torch.load(args.acts_path)
    loader = LlamaModelWrapper(args.model, device=args.device)
    model = loader.model
    logits_orig, _ = model.run_with_cache(
        args.prompt, reset_hooks_end=True, clear_contexts=True
    )
    orig_logits = logits_orig[0, -1, :]
    orig_idx = orig_logits.argmax().item()
    patcher = Patcher(model)
    hook_keys = [k for k in cache.keys() if k.startswith("blocks.")]
    patch_results = {}

    for key in hook_keys:
        patcher.clear()
        patcher.apply_patches(cache, [key])
        logits_p, _ = model.run_with_cache(
            args.prompt, reset_hooks_end=True, clear_contexts=True
        )
        patched_logits = logits_p[0, -1, :]
        delta = (patched_logits[orig_idx] - orig_logits[orig_idx]).item()
        patch_results[key] = delta

    torch.save(patch_results, args.out_path)
    print(f"[+] Saved patch results ({len(patch_results)} items) → {args.out_path}")

if __name__ == "__main__":
    main()
