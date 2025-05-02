import argparse, os, torch
from transformer_lens import HookedTransformer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    type=str,               required=True)
    p.add_argument("--device",   type=str,               default="cpu")
    p.add_argument("--prompt",   type=str,               required=True)
    p.add_argument("--layers",   type=int, nargs="+",     required=True)
    p.add_argument("--out_acts", type=str,               required=True)
    p.add_argument("--out_logits", type=str,             required=True)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_acts), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_logits), exist_ok=True)
    model = HookedTransformer.from_pretrained(
        args.model, device=args.device, fold_ln=False
    )
    def names_filter(name: str) -> bool:
        parts = name.split(".")
        return (len(parts) == 3
                and parts[0] == "blocks"
                and int(parts[1]) in args.layers
                and parts[2] == "resid_post")

    logits, cache = model.run_with_cache(
        args.prompt,
        names_filter=names_filter,
        remove_batch_dim=False,
        reset_hooks_end=True,
        clear_contexts=True
    )
    acts = {}
    logs = {}
    for l in args.layers:
        key = f"blocks.{l}.resid_post"
        acts[l] = cache[key][0].cpu()      # [seq_len, hidden_dim]
        logs[l] = logits.cpu()[0]         # [seq_len, vocab_size]

    torch.save(acts, args.out_acts)
    torch.save(logs, args.out_logits)
    print(f"[Capture] Residuals → {args.out_acts}, Logits → {args.out_logits}")

if __name__ == "__main__":
    main()
