from typing import Dict, Optional

def prune_top_k(importances: Dict[str, float], top_k: int) -> Dict[str, float]:
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_items[:top_k])

def prune_threshold(importances: Dict[str, float], threshold: float) -> Dict[str, float]:
    return {comp: score for comp, score in importances.items() if score >= threshold}
