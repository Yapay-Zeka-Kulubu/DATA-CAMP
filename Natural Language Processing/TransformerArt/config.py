"""
Hyperparameters and configuration for the GPT model
"""
import torch

# Training hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 128  # what is the maximum context length for predictions?
max_iters = 1000  # Total number of training iterations (reduced for demo)
eval_interval = 100  # How often to evaluate the model on train/val sets
eval_iters = 20  # Number of batches to evaluate the loss on
learning_rate = 3e-4  # learning rate for AdamW optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model Architecture Parameters (Small model for RTX 4060 8GB)
n_embd = 128  # Embedding dimension (hidden size) - reduced for faster training
n_head = 4  # Number of attention heads in multi-head self-attention
n_layer = 4  # Number of Transformer blocks stacked - reduced for demo
dropout = 0.1  # Dropout rate for regularization

# Data
data_file = 'nutuk.txt'

# Random seed for reproducibility
torch.manual_seed(1337)

print(f"Using device: {device}")
