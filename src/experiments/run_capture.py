import argparse
import os
import torch
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.model_loader.flexible_loader import FlexibleModelLoader
from src.activation_capture.capture import (
    ActivationCapture, ActivationCaptureConfig
)

def main():
    parser = argparse.ArgumentParser(
        description="Capture activations from transformer models for a given prompt"
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
        help="Input prompt to capture activations from"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Which layer indices to trace (0-based), default=all"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="outputs/activations/capture.pt",
        help="Where to save the activations (a .pt file)"
    )
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    
    print(f"[1/4] Loading model {args.model} on {args.device}...")
    try:
        loader = FlexibleModelLoader(args.model, device=args.device)
        model = loader.model
        tokenizer = loader.tokenizer
        
        # Print model info
        model_info = loader.get_model_info()
        print(f"Model loaded successfully:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print(f"[2/4] Starting activation capture (layers={args.layers})...")
    try:
        cfg = ActivationCaptureConfig(
            capture_residual=True,
            capture_mlp=True,
            capture_attention=False,
            capture_logits=True,
            layers_to_trace=args.layers
        )
        capturer = ActivationCapture(model, cfg)
        
        # Debug: print detected model structure
        capturer.print_model_structure()
        
        capturer.start()
    except Exception as e:
        print(f"Error setting up activation capture: {e}")
        return
    
    print(f"[3/4] Running forward on prompt: \"{args.prompt}\"")
    try:
        inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            _ = model(**inputs)
    except Exception as e:
        print(f"Error during forward pass: {e}")
        capturer.stop()
        return
    
    activations = capturer.get_activations()
    print(f"[4/4] Saving {len(activations)} activations to {args.out_path}")
    
    try:
        torch.save(activations, args.out_path)
        print(f"Activations saved successfully!")
        
        # Print activation shapes for debugging
        print("Activation shapes:")
        for key, value in activations.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
                
    except Exception as e:
        print(f"Error saving activations: {e}")
    
    capturer.stop()
    capturer.clear()
    print("Done.")

if __name__ == "__main__":
    main()
