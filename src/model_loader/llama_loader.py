# src/circuit_tracing_llama/model_loader/llama_loader.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LlamaModelWrapper:
    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        print(f"Loading tokenizer from {model_name_or_path} (fast)...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            legacy=False
        )

        print(f"Loading model from {model_name_or_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32
        ).to(device)
        self.model.eval()

        self.device = device

    def generate_text(self, prompt: str, max_new_tokens: int = 50) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
