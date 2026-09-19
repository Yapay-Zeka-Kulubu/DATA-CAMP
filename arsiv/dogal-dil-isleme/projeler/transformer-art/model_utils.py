"""
Utility functions for saving and loading models in HuggingFace format
"""
import torch
import json
import os
from pathlib import Path


def save_model_hf_format(model, data_loader, save_dir='model_output'):
    """
    Save model in HuggingFace-compatible format
    
    Args:
        model: trained GPT model
        data_loader: data loader with vocabulary
        save_dir: directory to save model files
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), save_path / 'pytorch_model.bin')
    
    # Save config
    config = {
        "model_type": "gpt",
        "vocab_size": data_loader.vocab_size,
        "n_embd": model.embeddings.token_embedding.embedding_dim,
        "n_head": model.blocks[0].attention.n_head,
        "n_layer": len(model.blocks),
        "block_size": model.block_size,
        "dropout": 0.0,  # Inference mode
        "architecture": "decoder-only-transformer"
    }
    
    with open(save_path / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # Save tokenizer (BPE)
    tokenizer_config = {
        "tokenizer_type": "bpe",
        "vocab_size": data_loader.vocab_size,
        "vocab": data_loader.tokenizer.vocab,
        "merges": data_loader.tokenizer.merges,
    }
    
    with open(save_path / 'tokenizer_config.json', 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    
    # Save README
    readme_content = f"""---
language: tr
tags:
- gpt
- transformer
- bpe
- turkish
license: mit
---

# GPT Model - Nutuk Dataset

Decoder-only transformer model trained on Atatürk's Nutuk (Turkish historical text).
Uses BPE (Byte Pair Encoding) tokenization implemented from scratch.

## Model Details

- **Model Type**: GPT (Decoder-only Transformer)
- **Language**: Turkish
- **Training Data**: Nutuk (~1.6M characters)
- **Tokenization**: BPE (Byte Pair Encoding)
- **Parameters**: ~{sum(p.numel() for p in model.parameters()):,}

## Architecture

- Embedding dimension: {config['n_embd']}
- Number of attention heads: {config['n_head']}
- Number of layers: {config['n_layer']}
- Context length: {config['block_size']}
- Vocabulary size: {config['vocab_size']} (BPE tokens)

## Usage

```python
from model.gpt import GPT
from bpe_tokenizer import BPETokenizer
import torch
import json

# Load config
with open('model_output/config.json', 'r') as f:
    config = json.load(f)

# Load tokenizer
with open('model_output/tokenizer_config.json', 'r') as f:
    tok_config = json.load(f)

tokenizer = BPETokenizer(vocab_size=config['vocab_size'])
tokenizer.vocab = tok_config['vocab']
tokenizer.merges = [tuple(m) for m in tok_config['merges']]
tokenizer.inverse_vocab = {{v: k for k, v in tokenizer.vocab.items()}}

# Create model
model = GPT(
    vocab_size=config['vocab_size'],
    n_embd=config['n_embd'],
    n_head=config['n_head'],
    n_layer=config['n_layer'],
    block_size=config['block_size'],
    dropout=0.0
)

# Load weights
model.load_state_dict(torch.load('model_output/pytorch_model.bin'))
model.eval()

# Generate text
# (see generate.py for full example)
```

## Training

- Optimizer: AdamW
- Learning rate: 3e-4
- Batch size: 32
- Context length: 128
- Tokenization: BPE with 500 vocab size

## Limitations

- BPE tokenization (subword level)
- Small model for demonstration purposes
- Trained on single historical text
"""
    
    with open(save_path / 'README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n✅ Model saved in HuggingFace format to: {save_path.absolute()}")
    print(f"   - pytorch_model.bin")
    print(f"   - config.json")
    print(f"   - tokenizer_config.json")
    print(f"   - README.md")


def load_model_hf_format(model_dir='model_output', device='cpu'):
    """
    Load model from HuggingFace format
    
    Args:
        model_dir: directory containing model files
        device: device to load model on
    
    Returns:
        model: loaded GPT model
        config: model configuration
        tokenizer_config: tokenizer configuration
    """
    from model.gpt import GPT
    
    model_path = Path(model_dir)
    
    # Load config
    with open(model_path / 'config.json', 'r') as f:
        config = json.load(f)
    
    # Load tokenizer config
    with open(model_path / 'tokenizer_config.json', 'r') as f:
        tokenizer_config = json.load(f)
    
    # Create model
    model = GPT(
        vocab_size=config['vocab_size'],
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        block_size=config['block_size'],
        dropout=0.0
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path / 'pytorch_model.bin', map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from: {model_path.absolute()}")
    
    return model, config, tokenizer_config
