"""
Transformer Block: combines attention, feed-forward, and normalization
"""
import torch.nn as nn
from components.attention import MultiHeadAttention
from components.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    Single Transformer block with:
    1. Masked Multi-Head Attention + Add & Norm
    2. Feed Forward + Add & Norm
    """
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.attention = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(n_embd, dropout)
        
        # Layer normalization (applied before attention and feed-forward)
        # This is "Pre-LN" variant which is more stable
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, n_embd) input tensor
        Returns:
            (B, T, n_embd) output tensor
        """
        # Attention block with residual connection
        # Pre-LN: normalize first, then apply attention, then add residual
        x = x + self.attention(self.ln1(x))
        
        # Feed-forward block with residual connection
        # Pre-LN: normalize first, then apply FFN, then add residual
        x = x + self.feed_forward(self.ln2(x))
        
        return x
