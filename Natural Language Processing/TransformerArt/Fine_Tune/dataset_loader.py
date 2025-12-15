"""
Dataset loader for GPT-4-Self-Instruct-Turkish
Prepares instruction-response pairs for fine-tuning
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
import torch
from Fine_Tune.ft_config import *
from bpe_tokenizer import BPETokenizer
import json


class InstructionDataset:
    """
    Load and prepare instruction-following dataset
    Format: <INST>instruction</INST><RESP>response</RESP>
    """
    
    def __init__(self, max_samples=1000):
        """
        Load dataset from HuggingFace
        
        Args:
            max_samples: maximum number of samples to use
        """
        print("\n" + "="*60)
        print("Loading Instruction Dataset")
        print("="*60)
        
        # Load dataset from HuggingFace
        print("Downloading from HuggingFace: CausalLM/GPT-4-Self-Instruct-Turkish")
        dataset = load_dataset("CausalLM/GPT-4-Self-Instruct-Turkish", split="train")
        
        # Limit samples
        if len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        
        print(f"Loaded {len(dataset)} samples")
        
        # Load pre-trained tokenizer
        print("\nLoading pre-trained BPE tokenizer...")
        self.tokenizer = BPETokenizer(vocab_size=500)
        tokenizer_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'bpe_tokenizer.json')
        self.tokenizer.load(tokenizer_path)
        
        # Add special tokens to vocabulary if not present
        special_tokens = [INSTRUCTION_START, INSTRUCTION_END, RESPONSE_START, RESPONSE_END]
        for token in special_tokens:
            if token not in self.tokenizer.vocab:
                self.tokenizer.vocab[token] = len(self.tokenizer.vocab)
        
        # Update inverse vocab
        self.tokenizer.inverse_vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        self.vocab_size = len(self.tokenizer.vocab)
        
        print(f"Vocabulary size (with special tokens): {self.vocab_size}")
        
        # Prepare instruction-response pairs
        self.data = []
        print("\nFormatting instruction-response pairs...")
        
        for i, example in enumerate(dataset):
            # Format: <INST>instruction</INST><RESP>response</RESP>
            # Try different possible field names
            instruction = example.get('instruction', example.get('input', example.get('prompt', '')))
            response = example.get('response', example.get('output', example.get('completion', '')))
            
            if instruction and response:
                formatted = f"{INSTRUCTION_START}{instruction}{INSTRUCTION_END}{RESPONSE_START}{response}{RESPONSE_END}"
                tokens = self.tokenizer.encode(formatted)
                if tokens:  # Only add if encoding succeeded
                    # Validate and clip token IDs to vocabulary range
                    valid_tokens = []
                    for token_id in tokens:
                        if token_id < self.vocab_size:
                            valid_tokens.append(token_id)
                        else:
                            # Replace out-of-range tokens with <UNK>
                            valid_tokens.append(self.tokenizer.vocab.get('<UNK>', 1))
                    self.data.extend(valid_tokens)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{len(dataset)} samples, Total tokens so far: {len(self.data):,}")
        
        # Convert to tensor
        self.data = torch.tensor(self.data, dtype=torch.long)
        
        # Final validation: ensure all tokens are within range
        if len(self.data) > 0:
            max_token = self.data.max().item()
            if max_token >= self.vocab_size:
                print(f"\n⚠️  Warning: Found tokens >= vocab_size ({max_token} >= {self.vocab_size})")
                print("Clipping to valid range...")
                self.data = torch.clamp(self.data, 0, self.vocab_size - 1)
        
        # Train/val split
        n = int(train_split * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total tokens: {len(self.data):,}")
        print(f"   Train tokens: {len(self.train_data):,}")
        print(f"   Validation tokens: {len(self.val_data):,}")
        print("="*60 + "\n")
    
    def get_batch(self, split):
        """
        Generate a batch of instruction-response sequences
        
        Args:
            split: 'train' or 'val'
        Returns:
            x, y: input and target tensors
        """
        data = self.train_data if split == 'train' else self.val_data
        
        # Random starting indices
        ix = torch.randint(len(data) - ft_block_size, (ft_batch_size,))
        
        # Create sequences
        x = torch.stack([data[i:i + ft_block_size] for i in ix])
        y = torch.stack([data[i + 1:i + ft_block_size + 1] for i in ix])
        
        # Move to device
        x, y = x.to(ft_device), y.to(ft_device)
        return x, y
    
    def encode(self, text):
        """Encode text to tokens"""
        return self.tokenizer.encode(text)
    
    def decode(self, tokens):
        """Decode tokens to text"""
        return self.tokenizer.decode(tokens)


# Test dataset loading
if __name__ == '__main__':
    dataset = InstructionDataset(max_samples=100)
    
    # Test batch
    x, y = dataset.get_batch('train')
    print(f"\nBatch shape: {x.shape}")
    print(f"Sample input: {dataset.decode(x[0].tolist()[:100])}")
