"""
Complete GPT Model: Decoder-only Transformer
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from components.embeddings import Embeddings
from model.transformer_block import TransformerBlock


class GPT(nn.Module):
    """
    Decoder-only Transformer (GPT-style) for character-level language modeling
    """
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.1):
        super().__init__()
        
        self.block_size = block_size
        
        # Input embeddings (token + position)
        self.embeddings = Embeddings(vocab_size, n_embd, block_size)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(n_embd)
        
        # Language modeling head (project to vocabulary)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT Model initialized with {n_params:,} parameters")
    
    def _init_weights(self, module):
        """Initialize weights with small random values"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        """
        Args:
            idx: (B, T) tensor of token indices
            targets: (B, T) tensor of target token indices (optional)
        Returns:
            logits: (B, T, vocab_size) predictions
            loss: scalar loss value (if targets provided)
        """
        B, T = idx.shape
        
        # Get embeddings
        x = self.embeddings(idx)  # (B, T, n_embd)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer normalization
        x = self.ln_f(x)  # (B, T, n_embd)
        
        # Project to vocabulary
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            # Reshape for cross-entropy loss
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Generate new tokens autoregressively
        
        Args:
            idx: (B, T) tensor of token indices (context)
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (higher = more random)
        Returns:
            (B, T+max_new_tokens) tensor of generated tokens
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Focus only on the last time step
            logits = logits[:, -1, :]  # (B, vocab_size)
            
            # Apply temperature
            logits = logits / temperature
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append to the sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        
        return idx
