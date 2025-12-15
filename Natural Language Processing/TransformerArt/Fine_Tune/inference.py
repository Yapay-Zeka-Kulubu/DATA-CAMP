"""
Interactive inference with fine-tuned instruction model
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from Fine_Tune.ft_config import *
from model.gpt import GPT
from bpe_tokenizer import BPETokenizer


def load_fine_tuned_model(checkpoint_path='fine_tuned_model.pt'):
    """Load fine-tuned model"""
    print("Loading fine-tuned model...")
    checkpoint = torch.load(checkpoint_path, map_location=ft_device)
    config = checkpoint['config']
    
    # Load tokenizer
    tokenizer = BPETokenizer(vocab_size=config['vocab_size'])
    tokenizer.vocab = checkpoint['tokenizer']['vocab']
    tokenizer.merges = [tuple(m) for m in checkpoint['tokenizer']['merges']]
    tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
    
    # Create model
    model = GPT(
        vocab_size=config['vocab_size'],
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        block_size=config['block_size'],
        dropout=0.0
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(ft_device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"   Vocabulary size: {config['vocab_size']}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer


def generate_response(model, tokenizer, instruction, max_tokens=200, temperature=0.7):
    """
    Generate response for an instruction
    
    Args:
        model: fine-tuned GPT model
        tokenizer: BPE tokenizer
        instruction: user instruction
        max_tokens: maximum tokens to generate
        temperature: sampling temperature
    """
    # Format prompt
    prompt = f"{INSTRUCTION_START}{instruction}{INSTRUCTION_END}{RESPONSE_START}"
    
    # Encode
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=ft_device)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=max_tokens, temperature=temperature)
    
    # Decode
    response = tokenizer.decode(generated[0].tolist())
    
    # Extract response part
    if RESPONSE_START in response:
        response = response.split(RESPONSE_START)[-1]
    if RESPONSE_END in response:
        response = response.split(RESPONSE_END)[0]
    
    return response.strip()


def interactive_mode():
    """Interactive instruction-following mode"""
    # Load model
    model, tokenizer = load_fine_tuned_model()
    
    print("\n" + "="*60)
    print("INSTRUCTION-FOLLOWING MODEL")
    print("="*60)
    print("Enter your instructions below.")
    print("Type 'quit' to exit\n")
    
    while True:
        instruction = input("📝 Instruction: ")
        
        if instruction.lower() in ['quit', 'exit', 'q']:
            break
        
        if not instruction.strip():
            continue
        
        # Get parameters
        try:
            max_tokens = int(input("   Max tokens (default 200): ") or "200")
            temperature = float(input("   Temperature (default 0.7): ") or "0.7")
        except ValueError:
            max_tokens = 200
            temperature = 0.7
        
        print("\n🤖 Generating response...\n")
        
        # Generate
        response = generate_response(model, tokenizer, instruction, max_tokens, temperature)
        
        print("-" * 60)
        print(response)
        print("-" * 60 + "\n")


if __name__ == '__main__':
    interactive_mode()
