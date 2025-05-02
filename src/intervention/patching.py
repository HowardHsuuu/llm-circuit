from transformer_lens import HookedTransformer
from torch import Tensor
from typing import Dict, List

class Patcher:
    def __init__(self, model: HookedTransformer):
        self.model = model
        self.edits = model.edit_hooks

    def apply_patches(self, cache: Dict[str, Tensor], hook_keys: List[str]):
        for key in hook_keys:
            tensor = cache[key]
            self.edits.add_patch(key, tensor)

    def clear(self):
        self.edits.clear_all()
