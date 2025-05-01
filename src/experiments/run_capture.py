import argparse
import os
import torch

from circuit_tracing_llama.model_loader.llama_loader import LlamaModelWrapper
from circuit_tracing_llama.activation_capture.capture import (
    ActivationCapture, ActivationCaptureConfig
)

def main():
    parser = argparse.ArgumentParser(
        description="Capture activations from LLaMA for a given prompt"
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
        "--out_path",
        type=str,
        default="outputs/activations/capture.pt",
        help=""
    )
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    print(f"[1/4] Loading model {args.model} on {args.device}...")
    loader = LlamaModelWrapper(args.model, device=args.device)
    model = loader.model
    print(f"[2/4] Starting activation capture (layers={args.layers})...")
    cfg = ActivationCaptureConfig(
        capture_residual=True,
        capture_mlp=True,
        capture_attention=False,
        capture_logits=True,
        layers_to_trace=args.layers
    )
    capturer = ActivationCapture(model, cfg)
    capturer.start()
    print(f"[3/4] Running forward on prompt: \"{args.prompt}\"")
    _ = loader.generate_text(args.prompt, max_new_tokens=0)
    activations = capturer.get_activations()
    print(f"[4/4] Saving {len(activations)} activation tensors to {args.out_path}...")
    torch.save(activations, args.out_path)

    capturer.stop()
    capturer.clear()
    print("Done.")

if __name__ == "__main__":
    main()
