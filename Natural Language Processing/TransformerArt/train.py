"""
Training script for the GPT model
"""
import torch
from config import *
from data_loader import data_loader
from model.gpt import GPT


@torch.no_grad()
def estimate_loss(model):
    """Estimate loss on train and validation sets"""
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = data_loader.get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out


def train():
    """Main training loop"""
    # Create model
    model = GPT(
        vocab_size=data_loader.vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout
    )
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    # Training loop
    for iter in range(max_iters):
        # Evaluate loss periodically
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"Step {iter:4d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Get batch
        xb, yb = data_loader.get_batch('train')
        
        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    
    # Save model in HuggingFace format
    from model_utils import save_model_hf_format
    save_model_hf_format(model, data_loader, save_dir='model_output')
    
    return model


if __name__ == '__main__':
    model = train()
    
    # Generate sample text
    print("\n" + "="*60)
    print("Generating sample text...")
    print("="*60 + "\n")
    
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500, temperature=0.8)
    text = data_loader.decode(generated[0].tolist())
    print(text)
