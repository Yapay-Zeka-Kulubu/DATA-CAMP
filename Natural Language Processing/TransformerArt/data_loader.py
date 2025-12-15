"""
Data loading and preprocessing with BPE tokenization
"""
import torch
from config import batch_size, block_size, device, data_file
from bpe_tokenizer import BPETokenizer


class DataLoaderBPE:
    def __init__(self, file_path, vocab_size=500):
        """Initialize the data loader with BPE tokenization"""
        # Read the text file
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        print("\n" + "="*60)
        print("Initializing Data Loader with BPE")
        print("="*60)
        print(f"Text length: {len(self.text)} characters")
        
        # Train BPE tokenizer
        self.tokenizer = BPETokenizer(vocab_size=vocab_size)
        self.tokenizer.train(self.text)
        
        self.vocab_size = len(self.tokenizer.vocab)
        
        # Encode the entire text
        self.data = torch.tensor(self.tokenizer.encode(self.text), dtype=torch.long)
        
        # Split into train and validation sets (90/10)
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]
        
        print(f"\n📊 Data Statistics:")
        print(f"   Vocabulary size: {self.vocab_size}")
        print(f"   Train tokens: {len(self.train_data):,}")
        print(f"   Validation tokens: {len(self.val_data):,}")
        print(f"   Total tokens: {len(self.data):,}")
        print("="*60 + "\n")
    
    def encode(self, text):
        """Encode text to token IDs"""
        return self.tokenizer.encode(text)
    
    def decode(self, token_ids):
        """Decode token IDs to text"""
        return self.tokenizer.decode(token_ids)
    
    def get_batch(self, split):
        """Generate a small batch of data of inputs x and targets y"""
        data = self.train_data if split == 'train' else self.val_data
        
        # Randomly select starting indices for sequences
        ix = torch.randint(len(data) - block_size, (batch_size,))
        
        # Create input and target sequences
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        
        # Move to device
        x, y = x.to(device), y.to(device)
        return x, y
    
    def save_tokenizer(self, filepath='bpe_tokenizer.json'):
        """Save the trained tokenizer"""
        self.tokenizer.save(filepath)
    
    def load_tokenizer(self, filepath='bpe_tokenizer.json'):
        """Load a trained tokenizer"""
        self.tokenizer.load(filepath)
        self.vocab_size = len(self.tokenizer.vocab)


# Create global data loader instance with BPE
print("\n🚀 Loading data with BPE tokenization...")
data_loader = DataLoaderBPE(data_file, vocab_size=500)
vocab_size = data_loader.vocab_size

# Save tokenizer for later use
data_loader.save_tokenizer('bpe_tokenizer.json')



if __name__ == "__main__":
    # Test encoding/decoding
    print("Hello, World!")
    print("\nTest Encoding/Decoding:")
    print("-" * 60)
    test_sentence = "Merhaba, dünya!"
    encoded = data_loader.encode(test_sentence)
    decoded = data_loader.decode(encoded)
    
    print(f"Original: {test_sentence}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")


