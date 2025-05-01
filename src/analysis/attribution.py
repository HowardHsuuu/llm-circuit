from typing import Dict

def compute_importances(patch_deltas: Dict[str, float]) -> Dict[str, float]:
    return {comp: abs(delta) for comp, delta in patch_deltas.items()}
