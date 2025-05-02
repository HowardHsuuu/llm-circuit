import argparse, os, torch
from src.model_loader.llama_loader import LlamaModelWrapper
from src.activation_capture.capture import (
    ActivationCapture, ActivationCaptureConfig
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--out_path", type=str,
                        default="outputs/activations/capture.pt")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    loader = LlamaModelWrapper(args.model, device=args.device)
    cfg = ActivationCaptureConfig(
        hook_names=["resid_post", "mlp_out", "attn_out", "ln_final"],
        layers_to_trace=args.layers,
        remove_batch_dim=True
    )
    capturer = ActivationCapture(loader.model, cfg)
    _, cache = capturer.run(args.prompt)
    torch.save(cache, args.out_path)
    print(f"[+] Saved {len(cache)} activations → {args.out_path}")

if __name__ == "__main__":
    main()
