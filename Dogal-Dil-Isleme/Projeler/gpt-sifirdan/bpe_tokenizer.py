"""
Byte Pair Encoding (BPE) Tokenizer - Implemented from scratch
Based on the original BPE algorithm for subword tokenization
"""
import re
import json
from collections import defaultdict, Counter


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer implementation from scratch
    
    BPE Algorithm:
    1. Start with vocabulary of individual characters
    2. Find the most frequent pair of adjacent tokens
    3. Merge this pair into a new token
    4. Repeat until desired vocabulary size
    """
    
    def __init__(self, vocab_size=500):
        """
        Initialize BPE tokenizer
        
        Args:
            vocab_size: target vocabulary size (including special tokens)
        """
        self.vocab_size = vocab_size
        self.vocab = {}  # token -> id
        self.inverse_vocab = {}  # id -> token
        self.merges = []  # list of merge operations
        # Simple pattern for word tokenization
        self.pattern = re.compile(r'\w+|\s+|[^\w\s]+', re.UNICODE)
        
    def get_stats(self, words):
        """
        Count frequency of adjacent token pairs
        
        Args:
            words: dict of {word: frequency}
        Returns:
            pairs: Counter of {(token1, token2): frequency}
        """
        pairs = defaultdict(int)
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs
    
    def merge_vocab(self, pair, words):
        """
        Merge the most frequent pair in vocabulary
        
        Args:
            pair: tuple of (token1, token2) to merge
            words: current word dictionary
        Returns:
            new_words: updated word dictionary with merged pair
        """
        new_words = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word in words:
            # Replace the pair with merged token
            new_word = word.replace(bigram, replacement)
            new_words[new_word] = words[word]
        
        return new_words
    
    def train(self, text):
        """
        Train BPE tokenizer on text
        
        Args:
            text: training text
        """
        print("\n" + "="*60)
        print("Training BPE Tokenizer")
        print("="*60)
        
        # Step 1: Pre-tokenize text into words
        words = re.findall(r'\S+', text.lower())
        
        # Step 2: Initialize vocabulary with characters
        # Add space marker to end of each word
        word_freqs = Counter(words)
        vocab_words = {' '.join(list(word) + ['</w>']): freq 
                      for word, freq in word_freqs.items()}
        
        # Get initial vocabulary (all characters)
        chars = set()
        for word in vocab_words.keys():
            chars.update(word.split())
        
        # Initialize vocab with special tokens and characters
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        for i, char in enumerate(sorted(chars)):
            self.vocab[char] = i + 4
        
        print(f"Initial vocabulary size: {len(self.vocab)}")
        print(f"Unique words: {len(word_freqs)}")
        
        # Step 3: Iteratively merge most frequent pairs
        num_merges = self.vocab_size - len(self.vocab)
        
        for i in range(num_merges):
            # Get pair statistics
            pairs = self.get_stats(vocab_words)
            
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge the pair
            vocab_words = self.merge_vocab(best_pair, vocab_words)
            
            # Record the merge
            self.merges.append(best_pair)
            
            # Add merged token to vocabulary
            new_token = ''.join(best_pair)
            self.vocab[new_token] = len(self.vocab)
            
            if (i + 1) % 50 == 0:
                print(f"Merge {i+1}/{num_merges}: {best_pair[0]} + {best_pair[1]} -> {new_token}")
        
        # Create inverse vocabulary
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        print(f"\n✅ BPE training complete!")
        print(f"Final vocabulary size: {len(self.vocab)}")
        print(f"Number of merges: {len(self.merges)}")
        print("="*60 + "\n")
    
    def tokenize_word(self, word):
        """
        Tokenize a single word using learned BPE merges
        
        Args:
            word: word to tokenize
        Returns:
            list of tokens
        """
        # Start with characters
        word = ' '.join(list(word.lower()) + ['</w>'])
        
        # Apply merges in order
        for pair in self.merges:
            bigram = ' '.join(pair)
            if bigram in word:
                word = word.replace(bigram, ''.join(pair))
        
        return word.split()
    
    def encode(self, text):
        """
        Encode text to token IDs
        
        Args:
            text: input text
        Returns:
            list of token IDs
        """
        # Pre-tokenize into words
        words = re.findall(r'\S+', text)
        
        # Tokenize each word
        tokens = []
        for word in words:
            word_tokens = self.tokenize_word(word)
            for token in word_tokens:
                # Use <UNK> for unknown tokens
                tokens.append(self.vocab.get(token, self.vocab['<UNK>']))
        
        return tokens
    
    def decode(self, token_ids):
        """
        Decode token IDs back to text
        
        Args:
            token_ids: list of token IDs
        Returns:
            decoded text
        """
        tokens = [self.inverse_vocab.get(id, '<UNK>') for id in token_ids]
        
        # Join tokens and remove word boundary markers
        text = ''.join(tokens).replace('</w>', ' ')
        
        return text.strip()
    
    def save(self, filepath):
        """Save tokenizer to file"""
        data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'vocab_size': self.vocab_size
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Tokenizer saved to {filepath}")
    
    def load(self, filepath):
        """Load tokenizer from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.merges = [tuple(merge) for merge in data['merges']]
        self.vocab_size = data['vocab_size']
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        print(f"✅ Tokenizer loaded from {filepath}")
        print(f"Vocabulary size: {len(self.vocab)}")


# Example usage and testing
if __name__ == '__main__':
    # Test BPE tokenizer
    test_text = """
    Merhaba dünya! Bu bir test metnidir. 
    Byte Pair Encoding algoritması çok güçlü bir tokenizasyon yöntemidir.
    """
    
    # Train tokenizer
    tokenizer = BPETokenizer(vocab_size=200)
    tokenizer.train(test_text)
    
    # Test encoding/decoding
    print("\nTest Encoding/Decoding:")
    print("-" * 60)
    test_sentence = "Merhaba, dünya!"
    encoded = tokenizer.encode(test_sentence)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {test_sentence}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")
