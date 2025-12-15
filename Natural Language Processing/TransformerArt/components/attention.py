"""
Multi-Head Self-Attention mechanism with causal masking
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with causal masking for decoder
    """
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        
        # Key, Query, Value projections for all heads (in a batch)
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        
        # Output projection
        self.proj = nn.Linear(n_embd, n_embd)
        
        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask to ensure attention only to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, T, n_embd) input tensor
        Returns:
            (B, T, n_embd) output tensor
        """
        B, T, C = x.shape
        
        # Calculate query, key, values for all heads in batch
        k = self.key(x)    # (B, T, n_embd)
        q = self.query(x)  # (B, T, n_embd)
        v = self.value(x)  # (B, T, n_embd)
        
        # Split into multiple heads
        # (B, T, n_embd) -> (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        
        # Compute attention scores ("affinities")
        # (B, n_head, T, head_size) @ (B, n_head, head_size, T) -> (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_size ** 0.5))
        
        # Apply causal mask (prevent attending to future tokens)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        
        # Normalize to get attention weights
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        # (B, n_head, T, T) @ (B, n_head, T, head_size) -> (B, n_head, T, head_size)
        y = att @ v
        
        # Reassemble all head outputs side by side
        # (B, n_head, T, head_size) -> (B, T, n_head, head_size) -> (B, T, n_embd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.proj(y))
        
        return y
