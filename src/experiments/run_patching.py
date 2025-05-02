import argparse, os, torch
from transformer_lens import HookedTransformer

def main():
    p = argparse.ArgumentParser(
        description="Patch activations back into the model and measure Δlogit"
    )
    p.add_argument("--model",     type=str, required=True)
    p.add_argument("--device",    type=str, default="cpu")
    p.add_argument("--prompt",    type=str, required=True)
    p.add_argument("--acts_path", type=str, required=True)
    p.add_argument(
        "--out_path",
        type=str,
        default="outputs/patching/patch_results.pt"
    )
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    cache: dict[str, torch.Tensor] = torch.load(args.acts_path)
    model = HookedTransformer.from_pretrained(
        args.model, device=args.device, fold_ln=False
    )
    logits_orig, _ = model.run_with_cache(
        args.prompt,
        reset_hooks_end=True,
        clear_contexts=True
    )
    top_idx = logits_orig[0, -1].argmax().item()
    results: dict[str, float] = {}
    for name, tensor in cache.items():
        if not any(s in name for s in ("resid_post", "mlp_out", "attn_out")):
            continue

        model.reset_hooks()
        def hook_fn(activation, hook=None, *, patched=tensor):
            p = patched.to(args.device)
            if p.dim() == activation.dim() - 1:
                p = p.unsqueeze(0)
            return p

        model.add_hook(name, hook_fn, dir="fwd")
        logits_p, _ = model.run_with_cache(
            args.prompt,
            reset_hooks_end=True,
            clear_contexts=True
        )
        delta = (logits_p[0, -1, top_idx] - logits_orig[0, -1, top_idx]).item()
        results[name] = delta

    torch.save(results, args.out_path)
    print(f"[+] Saved {len(results)} patch deltas → {args.out_path}")

if __name__ == "__main__":
    main()
