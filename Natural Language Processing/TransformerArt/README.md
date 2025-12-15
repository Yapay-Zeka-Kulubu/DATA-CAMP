# Transformer Architecture - GPT from Scratch

Bu proje, decoder-only transformer (GPT) mimarisini **tamamen sıfırdan** PyTorch ile kodlar. BPE (Byte Pair Encoding) tokenizer dahil her şey scratch'ten implement edilmiştir. Atatürk'ün Nutuk metni üzerinde eğitilir.

## 🎯 Özellikler

✅ **Tamamen Sıfırdan Kodlanmış:**
- BPE Tokenizer (Byte Pair Encoding)
- Multi-Head Self-Attention
- Positional Encoding
- Feed Forward Networks
- Transformer Blocks
- Complete GPT Model

✅ **GPU Desteği:** CUDA ile hızlı eğitim
✅ **HuggingFace Format:** Standart model kaydetme
✅ **Türkçe Metin Üretimi:** Nutuk veri seti

## 📁 Proje Yapısı

```
TransformerArt/
├── components/
│   ├── embeddings.py      # Token ve pozisyon gömmeleri
│   ├── attention.py       # Multi-head self-attention
│   └── feedforward.py     # Feed-forward network
├── model/
│   ├── transformer_block.py  # Transformer bloğu
│   └── gpt.py             # Tam GPT modeli
├── bpe_tokenizer.py       # BPE tokenizer (sıfırdan kodlanmış!)
├── data_loader.py         # Veri yükleme ve BPE tokenization
├── config.py              # Hiperparametreler
├── train.py               # Eğitim scripti
├── generate.py            # Metin üretimi
├── model_utils.py         # Model kaydetme/yükleme (HF format)
├── nutuk.txt              # Veri seti
└── requirements.txt       # Bağımlılıklar
```

## 🏗️ Mimari

### BPE Tokenizer (Byte Pair Encoding)

Karakter seviyesi yerine **subword tokenization** kullanıyoruz:

1. **Başlangıç:** Her karakter ayrı bir token
2. **İterasyon:** En sık görülen token çiftini birleştir
3. **Tekrar:** İstenen vocabulary boyutuna kadar devam et

**Avantajlar:**
- Daha küçük vocabulary
- Bilinmeyen kelimeler için daha iyi genelleme
- Daha verimli encoding

### Transformer Mimarisi

```
Input Tokens
  ↓
[BPE Tokenization]
  ↓
Token + Positional Embeddings
  ↓
┌─────────────────────┐
│ Transformer Block 1 │
│  - Attention        │
│  - Add & Norm       │
│  - Feed Forward     │
│  - Add & Norm       │
└─────────────────────┘
  ↓
┌─────────────────────┐
│ Transformer Block 2 │
└─────────────────────┘
  ↓
... (4 blocks total)
  ↓
Layer Normalization
  ↓
Linear (vocab projection)
  ↓
Softmax → Output Probabilities
```

### Hiperparametreler

```python
# Model Architecture
n_embd = 128        # Embedding dimension
n_head = 4          # Attention heads (128/4 = 32 per head)
n_layer = 4         # Transformer blocks
block_size = 128    # Context length
dropout = 0.1       # Dropout rate

# Training
batch_size = 32     # Batch size
max_iters = 1000    # Training iterations
learning_rate = 3e-4
vocab_size = 500    # BPE vocabulary size
```

## 🚀 Kurulum ve Kullanım

### 1. Sanal Ortam Oluştur

```bash
# Sanal ortam oluştur
python -m venv venv

# Aktif et (Windows)
.\venv\Scripts\activate

# Aktif et (Linux/Mac)
source venv/bin/activate
```

### 2. Bağımlılıkları Yükle

```bash
# PyTorch CUDA desteğiyle (GPU için)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# veya CPU için
pip install torch numpy
```

### 3. Modeli Eğit

```bash
# Eğitimi başlat
python train.py
```

**Eğitim Çıktısı:**
```
🚀 Loading data with BPE tokenization...
============================================================
Training BPE Tokenizer
============================================================
Initial vocabulary size: 107
Unique words: 89543
Merge 50/396: a n -> an
Merge 100/396: e r -> er
...
✅ BPE training complete!
Final vocabulary size: 500
Number of merges: 393
============================================================

Using device: cuda
GPT Model initialized with 836,199 parameters

============================================================
Starting training...
============================================================
Step    0: train loss 6.2145, val loss 6.2138
Step  100: train loss 3.8421, val loss 3.8567
Step  200: train loss 3.2156, val loss 3.2289
...
Step  999: train loss 2.1234, val loss 2.1456

============================================================
Training completed!
============================================================

✅ Model saved in HuggingFace format to: model_output/
   - pytorch_model.bin
   - config.json
   - tokenizer_config.json
   - README.md

============================================================
Generating sample text...
============================================================
[Üretilen metin burada görünecek]
```

### 4. Metin Üret

```bash
# İnteraktif metin üretimi
python generate.py
```

**Kullanım:**
```
Prompt: Türkiye Cumhuriyeti
Max tokens (default 500): 300
Temperature (default 0.8): 0.7

Generating...
------------------------------------------------------------
[Model tarafından üretilen metin]
------------------------------------------------------------
```

## 📊 Model Detayları

| Özellik | Değer |
|---------|-------|
| **Model Tipi** | Decoder-only Transformer (GPT) |
| **Tokenization** | BPE (Byte Pair Encoding) |
| **Vocabulary Size** | 500 tokens |
| **Parameters** | ~836K |
| **Model Size** | 3.6 MB |
| **Context Length** | 128 tokens |
| **Training Data** | Nutuk (~1.6M characters) |
| **GPU Memory** | ~500 MB |

## 🔍 BPE Tokenizer Detayları

### Nasıl Çalışır?

```python
from bpe_tokenizer import BPETokenizer

# Tokenizer oluştur ve eğit
tokenizer = BPETokenizer(vocab_size=500)
tokenizer.train(text)

# Encode
text = "Merhaba dünya"
tokens = tokenizer.encode(text)  # [45, 123, 67, 89, ...]

# Decode
decoded = tokenizer.decode(tokens)  # "merhaba dünya"
```

### BPE Algoritması

1. **Initialization:** Tüm karakterler vocabulary'de
2. **Iteration:**
   - En sık görülen token çiftini bul
   - Bu çifti yeni bir token olarak birleştir
   - Vocabulary'ye ekle
3. **Repeat:** Hedef vocabulary boyutuna kadar

**Örnek Merge İşlemleri:**
```
Initial: ['m', 'e', 'r', 'h', 'a', 'b', 'a']
Merge 1: 'a' + 'b' -> 'ab'
Result:  ['m', 'e', 'r', 'h', 'ab', 'a']
Merge 2: 'e' + 'r' -> 'er'
Result:  ['m', 'er', 'h', 'ab', 'a']
...
```

## 📚 Kod Yapısı

### BPE Tokenizer ([bpe_tokenizer.py](file:///c:/Users/w/Desktop/Kodlama/VsCode/HelloWorld/TransformerArt/bpe_tokenizer.py))

```python
class BPETokenizer:
    def train(text):        # BPE eğitimi
    def encode(text):       # Text -> token IDs
    def decode(ids):        # Token IDs -> text
    def save(filepath):     # Tokenizer kaydet
    def load(filepath):     # Tokenizer yükle
```

### Transformer Components

- **Embeddings:** Token + Positional embeddings
- **Attention:** Multi-head self-attention with causal masking
- **FeedForward:** 2-layer MLP with GELU
- **TransformerBlock:** Attention + FFN + Residual + LayerNorm
- **GPT:** Complete model assembly

## 🎓 Öğrenme Kaynakları

### Implemented Concepts

✅ **BPE Tokenization:**
- Subword segmentation
- Vocabulary learning
- Merge operations

✅ **Transformer Architecture:**
- Self-attention mechanism
- Multi-head attention
- Positional encoding
- Feed-forward networks
- Layer normalization
- Residual connections

✅ **Training:**
- AdamW optimizer
- Learning rate scheduling
- Gradient descent
- Loss calculation

## 🔧 Troubleshooting

### CUDA Hatası
```bash
# CUDA versiyonunu kontrol et
nvidia-smi

# Uygun PyTorch versiyonunu kur
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Memory Hatası
```python
# config.py'de batch_size'ı küçült
batch_size = 16  # 32 yerine
```

## 📝 Referanslar

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) - BPE Paper
- [Andrej Karpathy - nanoGPT](https://github.com/karpathy/nanoGPT)
- [noktali-virgul-ai-lectures](https://github.com/Cengineer00/noktali-virgul-ai-lectures)

## 📄 Lisans

MIT License - Eğitim amaçlı kullanım için serbesttir.

