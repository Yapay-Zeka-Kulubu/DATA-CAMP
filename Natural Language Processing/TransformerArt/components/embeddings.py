"""
Embedding layers: Token Embedding and Positional Encoding
"""
import torch
import torch.nn as nn


class Embeddings(nn.Module):
    """
    Combines token embeddings and positional embeddings
    """
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        # Token embedding: maps each token to a vector
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        
        # Positional embedding: learnable position encodings
        # Each position in the sequence gets its own embedding
        self.position_embedding = nn.Embedding(block_size, n_embd)
        
        self.block_size = block_size
    
    def forward(self, idx):
        """
        Args:
            idx: (B, T) tensor of token indices
        Returns:
            (B, T, n_embd) tensor of token + position embeddings
        """
        B, T = idx.shape
        
        # Get token embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        
        # Get position embeddings
        pos = torch.arange(T, device=idx.device)  # (T,)
        pos_emb = self.position_embedding(pos)  # (T, n_embd)
        
        # Add token and position embeddings
        x = tok_emb + pos_emb  # (B, T, n_embd)
        
        return x



if __name__ == "__main__":

    embed = Embeddings(vocab_size=500, n_embd=128, block_size=128)
    embed.forward(torch.randint(0, 500, (1, 128)))
    