from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from src.activation_capture.hooks import (
    register_residual_hook,
    register_mlp_hook,
    register_attn_hook,
    register_logits_hook,
)

@dataclass
class ActivationCaptureConfig:
    capture_residual: bool = True
    capture_mlp: bool = True
    capture_attention: bool = False
    capture_logits: bool = True
    layers_to_trace: Optional[List[int]] = None

class ActivationCapture:
    def __init__(self, model, config: ActivationCaptureConfig):
        self.model = model
        self.config = config
        self.activations: Dict[str, Any] = {}
        self._handles = []

    def _get_all_layer_idxs(self) -> List[int]:
        return list(range(len(self.model.transformer.h)))

    def start(self):
        layer_idxs = self.config.layers_to_trace or self._get_all_layer_idxs()

        if self.config.capture_residual:
            handles = register_residual_hook(self.model, layer_idxs, self.activations)
            self._handles.extend(handles)

        if self.config.capture_mlp:
            handles = register_mlp_hook(self.model, layer_idxs, self.activations)
            self._handles.extend(handles)

        if self.config.capture_attention:
            handles = register_attn_hook(self.model, layer_idxs, self.activations)
            self._handles.extend(handles)

        if self.config.capture_logits:
            handle = register_logits_hook(self.model, self.activations)
            self._handles.append(handle)

    def stop(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def clear(self):
        self.activations.clear()

    def get_activations(self) -> Dict[str, Any]:
        return self.activations

if __name__ == "__main__":
    # Example usage
    from src.model_loader.llama_loader import LlamaModelWrapper
    loader = LlamaModelWrapper("meta-llama/Llama-3.2-3B-Instruct", device="cuda")
    model = loader.model
    cfg = ActivationCaptureConfig(
        capture_residual=True,
        capture_mlp=True,
        capture_attention=False,
        capture_logits=True,
        layers_to_trace=[0, 1, 2]
    )
    capturer = ActivationCapture(model, cfg)
    capturer.start()
    prompt = "Hello world"
    loader.generate_text(prompt, max_new_tokens=0)
    acts = capturer.get_activations()
    print({k: v.shape for k, v in acts.items()})
    capturer.stop()
    capturer.clear()
