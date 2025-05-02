import argparse, os, torch
from transformer_lens import HookedTransformer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    type=str, required=True)
    p.add_argument("--device",   type=str, default="cpu")
    p.add_argument("--prompt",   type=str, required=True)
    p.add_argument("--out_path", type=str, default="outputs/activations/capture.pt")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    print(f"[1/2] Loading HookedTransformer {args.model} on {args.device}…")
    model = HookedTransformer.from_pretrained(
        args.model, device=args.device, fold_ln=False
    )

    def names_filter(name: str) -> bool:
        return any(
            name.endswith(suffix)
            for suffix in ("resid_post", "mlp_out", "attn_out")
        )

    print(f"[2/2] Running forward+cache for prompt “{args.prompt[:50]}…”")
    _, raw_cache = model.run_with_cache(
        args.prompt,
        names_filter=names_filter,
        remove_batch_dim=True,
        reset_hooks_end=True,
        clear_contexts=True
    )
    cache = {
        k: v.cpu().clone()
        for k, v in raw_cache.items()
        if isinstance(v, torch.Tensor)
    }
    torch.save(cache, args.out_path)
    print(f"[+] Saved {len(cache)} activations → {args.out_path}")

if __name__ == "__main__":
    main()
