from transformer_lens import HookedTransformer

class LlamaModelWrapper:
    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        print(f"Loading HookedTransformer from {model_name_or_path} on {device}...")
        # fold_ln=False to match original LLaMA behavior
        self.model = HookedTransformer.from_pretrained(
            model_name_or_path,
            device=device,
            fold_ln=False
        )
        self.device = device

    def run_with_cache(self, prompt: str):
        logits, cache = self.model.run_with_cache(
            prompt,
            reset_hooks_end=True,
            clear_contexts=True
        )
        return logits, cache
