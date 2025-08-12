import torch
from torch import nn
from typing import Dict, List, Optional, Tuple, Any
import re

class ModelStructureDetector:
    """Dynamically detects the structure of different transformer models."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.structure = self._detect_structure()
    
    def _detect_structure(self) -> Dict[str, Any]:
        """Detect the model structure by analyzing the model's module hierarchy."""
        structure = {
            'layers_path': None,
            'layer_type': None,
            'mlp_path': None,
            'attention_path': None,
            'lm_head_path': None,
            'embedding_path': None,
            'num_layers': 0
        }
        
        # Try to find transformer layers
        possible_layer_paths = [
            'layers',           # LLaMA, Mistral, etc.
            'transformer.h',    # GPT-2, BERT, etc.
            'transformer.layers', # Some T5 variants
            'encoder.layers',   # T5, BART, etc.
            'decoder.layers',   # T5 decoder
            'blocks',           # Some older models
        ]
        
        for path in possible_layer_paths:
            layers = self._get_nested_attribute(self.model, path)
            if layers is not None and len(layers) > 0:
                structure['layers_path'] = path
                structure['num_layers'] = len(layers)
                structure['layer_type'] = type(layers[0]).__name__
                break
        
        # Try to find MLP/FFN components
        if structure['layers_path']:
            sample_layer = self._get_nested_attribute(self.model, structure['layers_path'])[0]
            mlp_paths = ['mlp', 'feed_forward', 'ffn', 'feedforward']
            for path in mlp_paths:
                if hasattr(sample_layer, path):
                    structure['mlp_path'] = path
                    break
        
        # Try to find attention components
        if structure['layers_path']:
            sample_layer = self._get_nested_attribute(self.model, structure['layers_path'])[0]
            attn_paths = ['self_attn', 'attention', 'attn', 'self_attention']
            for path in attn_paths:
                if hasattr(sample_layer, path):
                    structure['attention_path'] = path
                    break
        
        # Find language model head
        head_paths = ['lm_head', 'head', 'classifier', 'output_projection']
        for path in head_paths:
            if hasattr(self.model, path):
                structure['lm_head_path'] = path
                break
        
        # Find embeddings
        embed_paths = ['embed_tokens', 'embeddings', 'embedding', 'word_embeddings']
        for path in embed_paths:
            if hasattr(self.model, path):
                structure['embedding_path'] = path
                break
        
        return structure
    
    def _get_nested_attribute(self, obj: Any, path: str) -> Optional[Any]:
        """Safely get nested attribute using dot notation."""
        try:
            for part in path.split('.'):
                obj = getattr(obj, part)
            return obj
        except (AttributeError, IndexError):
            return None
    
    def get_layer(self, layer_idx: int) -> Optional[nn.Module]:
        """Get a specific layer by index."""
        if self.structure['layers_path'] is None:
            return None
        layers = self._get_nested_attribute(self.model, self.structure['layers_path'])
        if layers is not None and 0 <= layer_idx < len(layers):
            return layers[layer_idx]
        return None
    
    def get_mlp(self, layer_idx: int) -> Optional[nn.Module]:
        """Get MLP component from a specific layer."""
        layer = self.get_layer(layer_idx)
        if layer is None or self.structure['mlp_path'] is None:
            return None
        return getattr(layer, self.structure['mlp_path'], None)
    
    def get_attention(self, layer_idx: int) -> Optional[nn.Module]:
        """Get attention component from a specific layer."""
        layer = self.get_layer(layer_idx)
        if layer is None or self.structure['attention_path'] is None:
            return None
        return getattr(layer, self.structure['attention_path'], None)
    
    def get_lm_head(self) -> Optional[nn.Module]:
        """Get the language model head."""
        if self.structure['lm_head_path'] is None:
            return None
        return getattr(self.model, self.structure['lm_head_path'], None)
    
    def get_embedding(self) -> Optional[nn.Module]:
        """Get the embedding layer."""
        if self.structure['embedding_path'] is None:
            return None
        return getattr(self.model, self.structure['embedding_path'], None)
    
    def get_all_layer_indices(self) -> List[int]:
        """Get all available layer indices."""
        if self.structure['num_layers'] == 0:
            return []
        return list(range(self.structure['num_layers']))
    
    def print_structure(self):
        """Print the detected model structure for debugging."""
        print("Detected Model Structure:")
        print(f"  Layers path: {self.structure['layers_path']}")
        print(f"  Number of layers: {self.structure['num_layers']}")
        print(f"  Layer type: {self.structure['layer_type']}")
        print(f"  MLP path: {self.structure['mlp_path']}")
        print(f"  Attention path: {self.structure['attention_path']}")
        print(f"  LM head path: {self.structure['lm_head_path']}")
        print(f"  Embedding path: {self.structure['embedding_path']}")
