import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel
from typing import Optional, Union, Dict, Any
import warnings

class FlexibleModelLoader:
    """A flexible model loader that can handle different types of transformer models."""
    
    def __init__(
        self, 
        model_name_or_path: str, 
        device: str = "cpu",
        model_type: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the model loader.
        
        Args:
            model_name_or_path: HuggingFace model ID or local path
            device: Device to load model on ('cpu', 'cuda', 'mps')
            model_type: Optional model type hint ('causal', 'seq2seq', 'encoder')
            **kwargs: Additional arguments passed to model loading
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.model_type = model_type
        
        print(f"Loading tokenizer from {model_name_or_path}...")
        self.tokenizer = self._load_tokenizer()
        
        print(f"Loading model from {model_name_or_path}...")
        self.model = self._load_model(**kwargs)
        
        # Detect model type if not specified
        if self.model_type is None:
            self.model_type = self._detect_model_type()
        
        print(f"Model type detected: {self.model_type}")
        print(f"Model loaded on device: {device}")
        
    def _load_tokenizer(self):
        """Load the tokenizer with fallback options."""
        try:
            return AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                use_fast=True,
                legacy=False
            )
        except Exception as e:
            print(f"Warning: Fast tokenizer failed, trying legacy: {e}")
            try:
                return AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    use_fast=False,
                    legacy=True
                )
            except Exception as e2:
                print(f"Error loading tokenizer: {e2}")
                raise
    
    def _load_model(self, **kwargs):
        """Load the model with appropriate configuration."""
        # Determine dtype based on device
        if self.device.startswith("cuda") and torch.cuda.is_available():
            dtype = torch.float16
        elif self.device == "mps" and torch.backends.mps.is_available():
            dtype = torch.float32  # MPS doesn't support float16 well
        else:
            dtype = torch.float32
        
        # Try to load as causal LM first
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=dtype,
                **kwargs
            )
            return model
        except Exception as e:
            print(f"Warning: Could not load as causal LM: {e}")
        
        # Try as seq2seq model
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=dtype,
                **kwargs
            )
            return model
        except Exception as e:
            print(f"Warning: Could not load as seq2seq LM: {e}")
        
        # Try as generic AutoModel (for encoder-only models)
        try:
            model = AutoModel.from_pretrained(
                self.model_name_or_path,
                torch_dtype=dtype,
                **kwargs
            )
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Move to device
        model = model.to(self.device)
        model.eval()
        return model
    
    def _detect_model_type(self) -> str:
        """Detect the type of model loaded."""
        if hasattr(self.model, 'generate'):
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'is_encoder_decoder'):
                if self.model.config.is_encoder_decoder:
                    return "seq2seq"
                else:
                    return "causal"
            else:
                return "causal"  # Assume causal if we can't determine
        else:
            return "encoder"
    
    def generate_text(
        self, 
        prompt: str, 
        max_new_tokens: int = 50,
        **kwargs
    ) -> str:
        """Generate text from a prompt."""
        if self.model_type == "causal":
            return self._generate_causal(prompt, max_new_tokens, **kwargs)
        elif self.model_type == "seq2seq":
            return self._generate_seq2seq(prompt, max_new_tokens, **kwargs)
        else:
            raise ValueError(f"Text generation not supported for model type: {self.model_type}")
    
    def _generate_causal(self, prompt: str, max_new_tokens: int, **kwargs) -> str:
        """Generate text using a causal language model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _generate_seq2seq(self, prompt: str, max_new_tokens: int, **kwargs) -> str:
        """Generate text using a seq2seq model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def forward_pass(self, prompt: str) -> Dict[str, Any]:
        """Run a forward pass and return the outputs."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            "model_name": self.model_name_or_path,
            "model_type": self.model_type,
            "device": self.device,
            "tokenizer_class": self.tokenizer.__class__.__name__,
            "model_class": self.model.__class__.__name__,
        }
        
        if hasattr(self.model, 'config'):
            config = self.model.config
            info.update({
                "vocab_size": getattr(config, 'vocab_size', 'Unknown'),
                "hidden_size": getattr(config, 'hidden_size', 'Unknown'),
                "num_layers": getattr(config, 'num_hidden_layers', 'Unknown'),
                "num_attention_heads": getattr(config, 'num_attention_heads', 'Unknown'),
            })
        
        return info
