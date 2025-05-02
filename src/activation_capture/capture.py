from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable
from transformer_lens import HookedTransformer

@dataclass
class ActivationCaptureConfig:
    hook_names: List[str] = ("resid_post", "mlp_out", "attn_out", "ln_final")
    layers_to_trace: Optional[List[int]] = None
    remove_batch_dim: bool = True

class ActivationCapture:
    def __init__(self, model: HookedTransformer, config: ActivationCaptureConfig):
        self.model = model
        self.config = config
        self.cache: Dict[str, Any] = {}

    def run(self, prompt: str):
        def names_filter(name: str) -> bool:
            parts = name.split(".")
            if len(parts) != 3 or parts[0] != "blocks":
                return False
            layer_idx = int(parts[1])
            hook_point = parts[2]
            if hook_point not in self.config.hook_names:
                return False
            if self.config.layers_to_trace is not None and layer_idx not in self.config.layers_to_trace:
                return False
            return True

        logits, cache = self.model.run_with_cache(
            prompt,
            names_filter=names_filter,
            remove_batch_dim=self.config.remove_batch_dim,
            reset_hooks_end=True,
            clear_contexts=True
        )
        self.cache = cache
        return logits, cache

    def get_activations(self) -> Dict[str, Any]:
        return self.cache
