# src/circuit_tracing_llama/model_loader/llama_loader.py

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

class LlamaModelLoader:
    def __init__(self, model_name_or_path: str, device: str = "cpu"):
        print(f"Loading tokenizer from {model_name_or_path}...")
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        print(f"Loading LLaMA model from {model_name_or_path} on {device}...")
        self.model = LlamaForCausalLM.from_pretrained(model_name_or_path)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    loader = LlamaModelLoader(model_name, device="cuda")
    test_prompt = "Hello, how are you today?"
    print("Prompt:", test_prompt)
    result = loader.generate(test_prompt, max_new_tokens=20)
    print("Generated:", result)
