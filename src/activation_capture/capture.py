from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from src.activation_capture.hooks import (
    register_residual_hook,
    register_mlp_hook,
    register_attn_hook,
    register_logits_hook,
)
from src.model_loader.model_structure import ModelStructureDetector

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
        self.structure_detector = ModelStructureDetector(model)

    def _get_all_layer_idxs(self) -> List[int]:
        return self.structure_detector.get_all_layer_indices()

    def start(self):
        layer_idxs = self.config.layers_to_trace or self._get_all_layer_idxs()
        
        if not layer_idxs:
            print("Warning: No layers found to trace")
            return

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
            try:
                handle = register_logits_hook(self.model, self.activations)
                self._handles.append(handle)
            except ValueError as e:
                print(f"Warning: Could not capture logits: {e}")

    def stop(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def clear(self):
        self.activations.clear()

    def get_activations(self) -> Dict[str, Any]:
        return self.activations

    def print_model_structure(self):
        """Print the detected model structure for debugging."""
        self.structure_detector.print_structure()

if __name__ == "__main__":
    print("ActivationCapture module loaded successfully")
