"""
Instruction Fine-Tuning Configuration
Fine-tune pre-trained GPT model on instruction-following tasks
"""
import torch
import os
# Fine-tuning hyperparameters
ft_batch_size = 8  # Smaller batch for instruction data
ft_block_size = 256  # Longer context for instructions
ft_max_iters = 500  # Fewer iterations for fine-tuning
ft_eval_interval = 50
ft_eval_iters = 10
ft_learning_rate = 1e-4  # Lower learning rate for fine-tuning
ft_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset parameters
max_samples = 1000  # Use only 1000 samples for demo
train_split = 0.9  # 90% train, 10% validation

# Model loading
pretrained_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model_output')

# Special tokens for instruction format
INSTRUCTION_START = "<INST>"
INSTRUCTION_END = "</INST>"
RESPONSE_START = "<RESP>"
RESPONSE_END = "</RESP>"

print(f"Fine-tuning device: {ft_device}")
