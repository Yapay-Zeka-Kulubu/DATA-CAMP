"""
Text generation script - load trained model and generate text
"""
import torch
from data_loader import data_loader
from model.gpt import GPT
from config import device


def load_model(model_dir='model_output'):
    """Load trained model from HuggingFace format"""
    from model_utils import load_model_hf_format
    
    model, config, tokenizer_config = load_model_hf_format(model_dir, device)
    return model


def generate_text(model, prompt="", max_tokens=500, temperature=0.8):
    """
    Generate text from the model
    
    Args:
        model: trained GPT model
        prompt: starting text (optional)
        max_tokens: number of tokens to generate
        temperature: sampling temperature (higher = more random)
    """
    # Encode prompt
    if prompt:
        context = torch.tensor([data_loader.encode(prompt)], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=max_tokens, temperature=temperature)
    
    # Decode and return
    return data_loader.decode(generated[0].tolist())


def interactive_generation(model):
    """Interactive text generation"""
    print("\n" + "="*60)
    print("Interactive Text Generation")
    print("="*60)
    print("Enter a prompt (or press Enter for random generation)")
    print("Type 'quit' to exit\n")
    
    while True:
        prompt = input("Prompt: ")
        
        if prompt.lower() == 'quit':
            break
        
        # Get generation parameters
        try:
            max_tokens = int(input("Max tokens (default 500): ") or "500")
            temperature = float(input("Temperature (default 0.8): ") or "0.8")
        except ValueError:
            print("Invalid input, using defaults")
            max_tokens = 500
            temperature = 0.8
        
        print("\nGenerating...\n")
        text = generate_text(model, prompt, max_tokens, temperature)
        print("-" * 60)
        print(text)
        print("-" * 60 + "\n")


if __name__ == '__main__':
    # Load model
    model = load_model()
    
    # Interactive generation
    interactive_generation(model)
