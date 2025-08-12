# llm-circuit

A flexible toolkit for analyzing and understanding the internal workings of Large Language Models (LLMs) through activation patching and circuit analysis.

## 🚀 **Key Features**

- **Universal Model Support**: Works with most open-source transformer models (LLaMA, GPT, BERT, T5, etc.)
- **Automatic Structure Detection**: Dynamically detects model architecture without hardcoded paths
- **Activation Capture**: Capture activations from different model components
- **Activation Patching**: Replace activations to understand causal relationships
- **Circuit Analysis**: Identify which components are critical for specific behaviors
- **CPU/GPU Support**: Runs on CPU, CUDA, or MPS devices

## 🏗️ **Architecture**

The project automatically detects and adapts to different transformer architectures:

- **Causal Language Models**: LLaMA, GPT, DialoGPT, etc.
- **Sequence-to-Sequence**: T5, BART, etc.
- **Encoder-only**: BERT, DistilBERT, etc.

## 📦 **Installation**

```bash
pip install -r requirements.txt
```

## 🔧 **Usage**

### 1. **Capture Activations**

Capture activations from any transformer model:

```bash
python src/experiments/run_capture.py \
    --model "microsoft/DialoGPT-medium" \
    --device "cpu" \
    --prompt "Your prompt here" \
    --out_path "outputs/activations/capture.pt"
```

### 2. **Run Patching Experiments**

Run activation patching to understand causal relationships:

```bash
python src/experiments/run_patching.py \
    --model "microsoft/DialoGPT-medium" \
    --device "cpu" \
    --prompt "Your prompt here" \
    --acts_path "outputs/activations/capture.pt" \
    --out_path "outputs/patching/results.pt"
```

### 3. **Analyze Results**

Analyze patching results and identify important components:

```bash
python src/experiments/run_analysis.py \
    --patch_path "outputs/patching/results.pt" \
    --top_k 10 \
    --out_importances "outputs/analysis/importances.pt" \
    --out_pruned "outputs/analysis/pruned.json"
```

## 🧪 **Testing**

Test the flexible implementation with different models:

```bash
python test_flexible.py
```

## 🔍 **How It Works**

### **Model Structure Detection**

The `ModelStructureDetector` automatically identifies:
- Transformer layer locations
- MLP/FFN components
- Attention mechanisms
- Language model heads
- Embedding layers

### **Activation Capture**

Uses PyTorch hooks to intercept activations at different points:
- **Residual connections**: Between transformer layers
- **MLP outputs**: Feed-forward network activations
- **Attention outputs**: Self-attention mechanism outputs
- **Logits**: Final model predictions

### **Activation Patching**

Systematically replaces activations to measure their causal impact:
1. Capture activations from one context
2. Patch them into another context
3. Measure changes in model outputs
4. Identify critical components

## 🎯 **Supported Models**

The toolkit automatically works with models from:
- **Meta**: LLaMA, LLaMA-2, LLaMA-3
- **Microsoft**: DialoGPT, Phi
- **Google**: T5, BERT, DistilBERT
- **Hugging Face**: Most models in the Hub
- **Custom**: Any model following standard transformer architecture

## 🚨 **Troubleshooting**

### **Model Not Supported**
- The toolkit automatically detects model structure
- If a model fails, check that it follows standard transformer architecture
- Use `capturer.print_model_structure()` to debug structure detection

### **Memory Issues**
- Use smaller models for testing (e.g., DialoGPT-medium instead of LLaMA-3)
- Run on CPU if GPU memory is insufficient
- Reduce the number of layers traced

### **Performance**
- CPU execution is slower but more accessible
- GPU acceleration recommended for larger models
- MPS support for Apple Silicon Macs

## 📚 **Examples**

### **Basic Usage**

```python
from src.model_loader.flexible_loader import FlexibleModelLoader
from src.activation_capture.capture import ActivationCapture, ActivationCaptureConfig

# Load any model
loader = FlexibleModelLoader("microsoft/DialoGPT-medium", device="cpu")

# Set up activation capture
cfg = ActivationCaptureConfig(
    capture_residual=True,
    capture_mlp=True,
    layers_to_trace=[0, 1, 2]
)

capturer = ActivationCapture(loader.model, cfg)
capturer.start()

# Run model
prompt = "Hello world"
inputs = loader.tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    _ = loader.model(**inputs)

# Get activations
activations = capturer.get_activations()
capturer.stop()
```

## 🤝 **Contributing**

The toolkit is designed to be extensible. To add support for new model types:

1. The `ModelStructureDetector` automatically handles most cases
2. Add specific detection logic in `_detect_structure()` if needed
3. Test with the provided test scripts

## 📄 **License**

This project is open source. Please check individual model licenses for compliance.