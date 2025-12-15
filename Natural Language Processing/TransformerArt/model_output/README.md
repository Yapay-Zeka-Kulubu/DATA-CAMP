---
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
- **Parameters**: ~938,228

## Architecture

- Embedding dimension: 128
- Number of attention heads: 4
- Number of layers: 4
- Context length: 128
- Vocabulary size: 500 (BPE tokens)

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
tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}

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
