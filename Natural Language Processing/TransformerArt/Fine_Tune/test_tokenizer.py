"""
Quick test script to debug tokenization issues
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bpe_tokenizer import BPETokenizer
from Fine_Tune.ft_config import *

# Load tokenizer
tokenizer = BPETokenizer(vocab_size=500)
tokenizer_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'bpe_tokenizer.json')
tokenizer.load(tokenizer_path)

print(f"Original vocab size: {len(tokenizer.vocab)}")

# Add special tokens
special_tokens = [INSTRUCTION_START, INSTRUCTION_END, RESPONSE_START, RESPONSE_END]
for token in special_tokens:
    if token not in tokenizer.vocab:
        tokenizer.vocab[token] = len(tokenizer.vocab)
        print(f"Added: {token} -> {tokenizer.vocab[token]}")

tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
print(f"New vocab size: {len(tokenizer.vocab)}")

# Test encoding
test_text = f"{INSTRUCTION_START}Merhaba{INSTRUCTION_END}{RESPONSE_START}Selam{RESPONSE_END}"
tokens = tokenizer.encode(test_text)
print(f"\nTest text: {test_text}")
print(f"Tokens: {tokens}")
print(f"Max token: {max(tokens) if tokens else 'N/A'}")
print(f"Min token: {min(tokens) if tokens else 'N/A'}")

# Check for out-of-range tokens
vocab_size = len(tokenizer.vocab)
out_of_range = [t for t in tokens if t >= vocab_size]
if out_of_range:
    print(f"\n⚠️  Out of range tokens: {out_of_range}")
else:
    print(f"\n✅ All tokens within range [0, {vocab_size-1}]")
