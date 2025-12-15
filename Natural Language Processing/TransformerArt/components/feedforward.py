"""
Feed Forward Network (MLP)
"""
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network
    Two-layer MLP with GELU activation
    """
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        # Expansion factor of 4 is standard in transformers
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand
            nn.GELU(),                       # Non-linearity
            nn.Linear(4 * n_embd, n_embd),  # Project back
            nn.Dropout(dropout),             # Regularization
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, T, n_embd) input tensor
        Returns:
            (B, T, n_embd) output tensor
        """
        return self.net(x)
