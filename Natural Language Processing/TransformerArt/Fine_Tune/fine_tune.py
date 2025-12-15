"""
Fine-tuning script for instruction-following
Loads pre-trained model and fine-tunes on instruction dataset
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from Fine_Tune.ft_config import *
from Fine_Tune.dataset_loader import InstructionDataset
from model.gpt import GPT
from model_utils import load_model_hf_format


@torch.no_grad()
def estimate_loss(model, dataset):
    """Estimate loss on train and validation sets"""
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(ft_eval_iters)
        for k in range(ft_eval_iters):
            X, Y = dataset.get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out


def fine_tune():
    """Main fine-tuning loop"""
    print("\n" + "="*60)
    print("INSTRUCTION FINE-TUNING")
    print("="*60)
    
    # Load instruction dataset
    dataset = InstructionDataset(max_samples=max_samples)
    
    # Load pre-trained model
    print("\nLoading pre-trained model...")
    model, config, tokenizer_config = load_model_hf_format(pretrained_model_path, ft_device)
    
    # Update model for new vocabulary size (with special tokens)
    if dataset.vocab_size != config['vocab_size']:
        print(f"\n⚠️  Vocabulary size changed: {config['vocab_size']} -> {dataset.vocab_size}")
        print("Expanding embedding layer...")
        
        # Expand token embedding
        old_embeddings = model.embeddings.token_embedding.weight.data
        new_embeddings = torch.nn.Embedding(dataset.vocab_size, config['n_embd'])
        new_embeddings.weight.data[:config['vocab_size']] = old_embeddings
        # Initialize new tokens with small random values
        new_embeddings.weight.data[config['vocab_size']:] = torch.randn(
            dataset.vocab_size - config['vocab_size'], 
            config['n_embd']
        ) * 0.02
        model.embeddings.token_embedding = new_embeddings
        
        # Expand output layer
        old_lm_head = model.lm_head.weight.data
        new_lm_head = torch.nn.Linear(config['n_embd'], dataset.vocab_size)
        new_lm_head.weight.data[:config['vocab_size']] = old_lm_head
        new_lm_head.weight.data[config['vocab_size']:] = torch.randn(
            dataset.vocab_size - config['vocab_size'],
            config['n_embd']
        ) * 0.02
        model.lm_head = new_lm_head
        
        model = model.to(ft_device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    
    # Create optimizer (lower learning rate for fine-tuning)
    optimizer = torch.optim.AdamW(model.parameters(), lr=ft_learning_rate)
    
    print("\n" + "="*60)
    print("Starting fine-tuning...")
    print("="*60)
    
    # Fine-tuning loop
    for iter in range(ft_max_iters):
        # Evaluate periodically
        if iter % ft_eval_interval == 0 or iter == ft_max_iters - 1:
            losses = estimate_loss(model, dataset)
            print(f"Step {iter:4d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Get batch
        xb, yb = dataset.get_batch('train')
        
        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    print("\n" + "="*60)
    print("Fine-tuning completed!")
    print("="*60)
    
    # Save fine-tuned model
    print("\nSaving fine-tuned model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': dataset.vocab_size,
            'n_embd': config['n_embd'],
            'n_head': config['n_head'],
            'n_layer': config['n_layer'],
            'block_size': ft_block_size,
            'dropout': 0.0
        },
        'tokenizer': {
            'vocab': dataset.tokenizer.vocab,
            'merges': dataset.tokenizer.merges,
        }
    }, 'fine_tuned_model.pt')
    
    print("✅ Fine-tuned model saved to: fine_tuned_model.pt")
    
    return model, dataset


if __name__ == '__main__':
    model, dataset = fine_tune()
    
    # Test generation
    print("\n" + "="*60)
    print("Testing instruction following...")
    print("="*60)
    
    # Test instruction
    test_instruction = "Türkiye'nin başkenti neresidir?"
    prompt = f"{INSTRUCTION_START}{test_instruction}{INSTRUCTION_END}{RESPONSE_START}"
    
    print(f"\nInstruction: {test_instruction}")
    print("Response: ", end="")
    
    # Encode and generate
    context = torch.tensor([dataset.encode(prompt)], dtype=torch.long, device=ft_device)
    generated = model.generate(context, max_new_tokens=100, temperature=0.7)
    response = dataset.decode(generated[0].tolist())
    
    # Extract response part
    if RESPONSE_START in response:
        response = response.split(RESPONSE_START)[-1]
    if RESPONSE_END in response:
        response = response.split(RESPONSE_END)[0]
    
    print(response)
