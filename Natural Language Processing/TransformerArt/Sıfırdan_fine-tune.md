# 🎓 Sıfırdan Fine-Tuning'e

**Kapsamlı Rehber: Teori + Kod + Matematik**

Bu doküman, transformer mimarisini sıfırdan anlamanız ve kendi modelinizi eğitmeniz için hazırlanmıştır. Her adımda hem teorik açıklama hem kod örneği bulacaksınız.

---

## 📚 İçindekiler

1. [Transformer Mimarisi Genel Bakış](#1-transformer-mimarisi-genel-bakış)
2. [BPE Tokenization](#2-bpe-tokenization)
3. [Embedding Katmanları](#3-embedding-katmanları)
4. [Multi-Head Self-Attention](#4-multi-head-self-attention)
5. [Feed-Forward Network](#5-feed-forward-network)
6. [Transformer Block](#6-transformer-block)
7. [Complete GPT Model](#7-complete-gpt-model)
8. [Pre-Training](#8-pre-training)
9. [Fine-Tuning](#9-fine-tuning)
10. [Önemli Parametreler](#10-önemli-parametreler)
11. [Debugging ve İpuçları](#11-debugging-ve-i̇puçları)

---

## 1. Transformer Mimarisi Genel Bakış

### 🎯 Ne Yapıyoruz?

Decoder-only transformer (GPT-style) modeli oluşturuyoruz. Bu model:
- Metni token'lara ayırır (BPE)
- Her token'ı vektöre dönüştürür (Embedding)
- Self-attention ile token'lar arası ilişkileri öğrenir
- Bir sonraki token'ı tahmin eder (Language Modeling)

### 📊 Mimari Diyagram

![Transformer Architecture](file:///C:/Users/w/.gemini/antigravity/brain/5a4c8119-18b9-427e-97c7-36b7d3551dd1/uploaded_image_1765724093511.png)

**Sağ Taraf (Decoder - Bizim Modelimiz):**
```
Input (Outputs shifted right)
    ↓
Output Embedding
    ↓
Positional Encoding
    ↓
┌─────────────────────────┐
│ Masked Multi-Head       │  ← Gelecek token'ları görmez
│ Attention               │
└─────────────────────────┘
    ↓
Add & Norm (Residual)
    ↓
┌─────────────────────────┐
│ Feed Forward            │
└─────────────────────────┘
    ↓
Add & Norm (Residual)
    ↓
(N× tekrar)
    ↓
Linear
    ↓
Softmax
    ↓
Output Probabilities
```

### 🔑 Temel Kavramlar

**1. Autoregressive (Otoregresif):**
- Model bir sonraki token'ı tahmin eder
- Sadece geçmiş token'lara bakar (causal masking)

**2. Decoder-Only:**
- Sadece decoder kısmını kullanıyoruz
- Encoder yok (BERT gibi modellerde var)

**3. Self-Attention:**
- Her token diğer token'larla ilişkisini öğrenir
- "Türkiye'nin başkenti" → "Ankara" ilişkisi

---

## 2. BPE Tokenization

### 📖 Teori

**Byte Pair Encoding (BPE)**, metni subword'lere ayırır.

**Neden Character-level değil?**
- ✅ Daha küçük vocabulary
- ✅ Bilinmeyen kelimeler için daha iyi
- ✅ Daha verimli

**Algoritma:**
```
1. Başlangıç: Her karakter ayrı token
   "merhaba" → ['m', 'e', 'r', 'h', 'a', 'b', 'a']

2. En sık görülen çifti birleştir
   'a' + 'b' çok sık → 'ab'
   "merhaba" → ['m', 'e', 'r', 'h', 'ab', 'a']

3. Tekrar et
   'e' + 'r' → 'er'
   "merhaba" → ['m', 'er', 'h', 'ab', 'a']

4. Hedef vocabulary boyutuna kadar devam
```

### 💻 Kod İncelemesi

**Dosya:** [`bpe_tokenizer.py`](file:///c:/Users/w/Desktop/Kodlama/VsCode/HelloWorld/TransformerArt/bpe_tokenizer.py)

```python
class BPETokenizer:
    def train(self, text):
        # 1. Metni kelimelere ayır
        words = re.findall(r'\S+', text.lower())
        
        # 2. Her kelimeyi karakterlere ayır + </w> ekle
        vocab_words = {' '.join(list(word) + ['</w>']): freq 
                      for word, freq in word_freqs.items()}
        
        # 3. İteratif merge
        for i in range(num_merges):
            # En sık çifti bul
            pairs = self.get_stats(vocab_words)
            best_pair = max(pairs, key=pairs.get)
            
            # Merge yap
            vocab_words = self.merge_vocab(best_pair, vocab_words)
            self.merges.append(best_pair)
```

**Örnek Çıktı:**
```
Initial vocabulary: 76 characters
Merge 50/424:  a + n → an
Merge 100/424: e + r → er
Merge 200/424: l + a → la
Final vocabulary: 500 tokens
```

### 🎯 Önemli Parametreler

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `vocab_size` | 500 | Hedef vocabulary boyutu |
| `num_merges` | 424 | Yapılacak merge sayısı |

**Trade-off:**
- Küçük vocab → Daha uzun sequence'ler
- Büyük vocab → Daha kısa sequence'ler ama daha fazla parametre

---

## 3. Embedding Katmanları

### 📖 Teori

**Token Embedding:**
- Her token'ı dense vector'e dönüştürür
- Öğrenilebilir (learnable)

**Positional Encoding:**
- Token'ın dizideki pozisyonunu kodlar
- Transformer'da sıra bilgisi yok, bu yüzden gerekli

### 🧮 Matematik

**Token Embedding:**
```
E_token ∈ ℝ^(vocab_size × n_embd)
token_id → E_token[token_id] ∈ ℝ^n_embd
```

**Positional Embedding:**
```
E_pos ∈ ℝ^(block_size × n_embd)
position → E_pos[position] ∈ ℝ^n_embd
```

**Final Embedding:**
```
x = E_token[token_id] + E_pos[position]
```

### 💻 Kod İncelemesi

**Dosya:** [`components/embeddings.py`](file:///c:/Users/w/Desktop/Kodlama/VsCode/HelloWorld/TransformerArt/components/embeddings.py)

```python
class Embeddings(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size):
        # Token embedding tablosu
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        
        # Positional embedding tablosu (learnable)
        self.position_embedding = nn.Embedding(block_size, n_embd)
    
    def forward(self, idx):
        B, T = idx.shape  # Batch, Time
        
        # Token embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        
        # Position embeddings
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos)  # (T, n_embd)
        
        # Topla
        x = tok_emb + pos_emb  # Broadcasting: (B,T,n_embd) + (T,n_embd)
        return x
```

**Örnek:**
```python
# Input: [45, 123, 67, 89]  (4 token)
# vocab_size=500, n_embd=128

tok_emb = [[0.12, -0.34, ...],  # 128 dim
           [0.56, 0.78, ...],
           [-0.23, 0.45, ...],
           [0.89, -0.12, ...]]

pos_emb = [[0.01, 0.02, ...],   # Position 0
           [0.03, 0.04, ...],   # Position 1
           [0.05, 0.06, ...],   # Position 2
           [0.07, 0.08, ...]]   # Position 3

x = tok_emb + pos_emb  # Element-wise addition
```

### 🎯 Önemli Parametreler

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `n_embd` | 128 | Embedding dimension |
| `vocab_size` | 500 | Vocabulary boyutu |
| `block_size` | 128 | Maximum sequence length |

**Parametre Sayısı:**
```
Token Embedding: vocab_size × n_embd = 500 × 128 = 64,000
Position Embedding: block_size × n_embd = 128 × 128 = 16,384
Toplam: 80,384 parametre
```

---

## 4. Multi-Head Self-Attention

### 📖 Teori

**Self-Attention**, her token'ın diğer token'larla ilişkisini öğrenir.

**Soru:** "Türkiye'nin başkenti Ankara'dır" cümlesinde "başkenti" kelimesi hangi kelimelere dikkat etmeli?
**Cevap:** "Türkiye'nin" ve "Ankara'dır" → İlişki öğrenilir!

### 🧮 Matematik

**Scaled Dot-Product Attention:**

```
Q = X × W_Q    (Query)
K = X × W_K    (Key)
V = X × W_V    (Value)

Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

**Adım adım:**

1. **Query, Key, Value hesapla:**
   ```
   Q, K, V ∈ ℝ^(T × d_k)
   d_k = n_embd / n_head
   ```

2. **Attention scores:**
   ```
   scores = Q × K^T ∈ ℝ^(T × T)
   scores[i,j] = "token i, token j'ye ne kadar dikkat ediyor?"
   ```

3. **Scaling:**
   ```
   scores = scores / √d_k
   ```
   Neden? Büyük d_k'da gradient'ler çok küçük olur.

4. **Causal Masking (Decoder için):**
   ```
   mask[i,j] = 0 if j > i else 1
   scores = scores.masked_fill(mask == 0, -inf)
   ```
   Token i, sadece i'den önceki token'lara bakabilir!

5. **Softmax:**
   ```
   attention_weights = softmax(scores)  # Her satır toplamı 1
   ```

6. **Weighted sum:**
   ```
   output = attention_weights × V
   ```

**Multi-Head:**
- Attention'ı paralel olarak n_head kez yap
- Her head farklı ilişkileri öğrenir
- Sonuçları concatenate et

### 💻 Kod İncelemesi

**Dosya:** [`components/attention.py`](file:///c:/Users/w/Desktop/Kodlama/VsCode/HelloWorld/TransformerArt/components/attention.py)

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        self.n_head = n_head
        self.head_size = n_embd // n_head  # Her head'in boyutu
        
        # Q, K, V projections
        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        
        # Output projection
        self.proj = nn.Linear(n_embd, n_embd)
        
        # Causal mask (lower triangular)
        self.register_buffer("mask",
            torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape  # Batch, Time, Channels
        
        # 1. Q, K, V hesapla
        q = self.query(x)  # (B, T, n_embd)
        k = self.key(x)
        v = self.value(x)
        
        # 2. Multi-head için reshape
        # (B, T, n_embd) → (B, T, n_head, head_size) → (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        
        # 3. Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_size ** 0.5))
        # (B, n_head, T, head_size) @ (B, n_head, head_size, T) 
        # = (B, n_head, T, T)
        
        # 4. Causal masking
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        
        # 5. Softmax
        att = F.softmax(att, dim=-1)  # Her satır toplamı 1
        
        # 6. Weighted sum
        y = att @ v  # (B, n_head, T, T) @ (B, n_head, T, head_size)
                     # = (B, n_head, T, head_size)
        
        # 7. Concatenate heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 8. Output projection
        y = self.proj(y)
        return y
```

**Görsel Örnek:**

```
Input: "Türkiye'nin başkenti"
Tokens: [T1, T2, T3]

Attention Matrix (after softmax):
       T1    T2    T3
T1  [ 1.0   0     0  ]  ← T1 sadece kendine bakar
T2  [ 0.3  0.7   0  ]  ← T2, T1'e %30, kendine %70
T3  [ 0.2  0.5  0.3 ]  ← T3 hepsine bakabilir

Causal Mask (üst üçgen -inf):
       T1    T2    T3
T1  [ OK   -∞   -∞  ]
T2  [ OK   OK   -∞  ]
T3  [ OK   OK   OK  ]
```

### 🎯 Önemli Parametreler

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `n_head` | 4 | Attention head sayısı |
| `head_size` | 32 | Her head boyutu (128/4) |
| `dropout` | 0.1 | Regularization |

**Parametre Sayısı (bir head için):**
```
W_Q: n_embd × n_embd = 128 × 128 = 16,384
W_K: n_embd × n_embd = 128 × 128 = 16,384
W_V: n_embd × n_embd = 128 × 128 = 16,384
W_O: n_embd × n_embd = 128 × 128 = 16,384
Toplam: 65,536 parametre
```

---

## 5. Feed-Forward Network

### 📖 Teori

**Feed-Forward Network (FFN)**, her pozisyona ayrı ayrı uygulanır.

**Amaç:**
- Non-linearity eklemek
- Representation capacity artırmak

### 🧮 Matematik

```
FFN(x) = GELU(x × W_1 + b_1) × W_2 + b_2

W_1 ∈ ℝ^(n_embd × 4*n_embd)  # Expansion
W_2 ∈ ℝ^(4*n_embd × n_embd)  # Projection
```

**GELU (Gaussian Error Linear Unit):**
```
GELU(x) = x × Φ(x)
Φ(x) = cumulative distribution function of standard normal
```

Neden GELU? ReLU'dan daha smooth, gradient flow daha iyi.

### 💻 Kod İncelemesi

**Dosya:** [`components/feedforward.py`](file:///c:/Users/w/Desktop/Kodlama/VsCode/HelloWorld/TransformerArt/components/feedforward.py)

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand: 128 → 512
            nn.GELU(),                       # Non-linearity
            nn.Linear(4 * n_embd, n_embd),  # Project: 512 → 128
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)  # (B, T, n_embd) → (B, T, n_embd)
```

**Neden 4× expansion?**
- Transformer paper'da standart
- Daha fazla capacity
- Trade-off: Parametre sayısı vs performance

### 🎯 Önemli Parametreler

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `expansion` | 4 | İç katman boyutu çarpanı |
| `dropout` | 0.1 | Regularization |

**Parametre Sayısı:**
```
W_1: n_embd × 4*n_embd = 128 × 512 = 65,536
b_1: 4*n_embd = 512
W_2: 4*n_embd × n_embd = 512 × 128 = 65,536
b_2: n_embd = 128
Toplam: 131,712 parametre
```

---

## 6. Transformer Block

### 📖 Teori

**Transformer Block**, attention ve FFN'i birleştirir.

**Önemli:** Residual connections ve Layer Normalization!

### 🧮 Matematik

```
# Pre-LN (Pre-Layer Normalization) variant
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

**Neden Residual?**
- Gradient flow iyileşir
- Derin network'ler eğitilebilir
- Identity mapping öğrenilebilir

**Neden LayerNorm?**
- Training stabilizasyonu
- Batch'e bağımsız (BatchNorm'dan farklı)

### 💻 Kod İncelemesi

**Dosya:** [`model/transformer_block.py`](file:///c:/Users/w/Desktop/Kodlama/VsCode/HelloWorld/TransformerArt/model/transformer_block.py)

```python
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.feed_forward = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)  # Pre-attention
        self.ln2 = nn.LayerNorm(n_embd)  # Pre-FFN
    
    def forward(self, x):
        # Attention block + residual
        x = x + self.attention(self.ln1(x))
        
        # FFN block + residual
        x = x + self.feed_forward(self.ln2(x))
        
        return x
```

**Görsel:**
```
Input x
  ↓
LayerNorm
  ↓
Attention ──┐
  ↓         │
  + ←───────┘  (Residual)
  ↓
LayerNorm
  ↓
FFN ───────┐
  ↓        │
  + ←──────┘  (Residual)
  ↓
Output
```

### 🎯 Parametre Sayısı

```
Attention: 65,536
FFN: 131,712
LayerNorm1: 2 × n_embd = 256
LayerNorm2: 2 × n_embd = 256
Toplam (1 block): 197,760 parametre
```

---

## 7. Complete GPT Model

### 📖 Teori

**GPT Model**, tüm bileşenleri birleştirir:
1. Embeddings
2. N× Transformer Blocks
3. Final LayerNorm
4. Language Modeling Head

### 💻 Kod İncelemesi

**Dosya:** [`model/gpt.py`](file:///c:/Users/w/Desktop/Kodlama/VsCode/HelloWorld/TransformerArt/model/gpt.py)

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.1):
        super().__init__()
        
        # 1. Embeddings
        self.embeddings = Embeddings(vocab_size, n_embd, block_size)
        
        # 2. Transformer blocks (N×)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        
        # 3. Final LayerNorm
        self.ln_f = nn.LayerNorm(n_embd)
        
        # 4. Language Modeling Head
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        # Embeddings
        x = self.embeddings(idx)  # (B, T, n_embd)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final norm
        x = self.ln_f(x)
        
        # Logits
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Loss (if targets provided)
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            return logits, loss
        
        return logits, None
```

### 🎯 Model Parametreleri

**Konfigürasyon:**
```python
vocab_size = 500
n_embd = 128
n_head = 4
n_layer = 4
block_size = 128
```

**Toplam Parametre:**
```
Embeddings: 80,384
Transformer Blocks: 197,760 × 4 = 791,040
Final LayerNorm: 256
LM Head: vocab_size × n_embd = 500 × 128 = 64,000
──────────────────────────────────────────
Toplam: 935,680 parametre (~936K)
```

---

## 8. Pre-Training

### 📖 Teori

**Pre-training**, modeli genel dil bilgisi öğretmek için yapılır.

**Amaç:**
- Türkçe dilbilgisi öğren
- Kelime ilişkilerini öğren
- Genel representation öğren

**Dataset:** Nutuk (~1.6M karakter)

### 🧮 Loss Function

**Cross-Entropy Loss:**
```
L = -∑ y_true × log(y_pred)

y_true: One-hot encoded gerçek token
y_pred: Model'in tahmin ettiği probability distribution
```

**Örnek:**
```
Gerçek token: "Ankara" (ID: 45)
Model tahminleri:
  Token 44: 0.1
  Token 45: 0.7  ← Doğru
  Token 46: 0.2

Loss = -log(0.7) = 0.357
```

### 💻 Training Loop

**Dosya:** [`train.py`](file:///c:/Users/w/Desktop/Kodlama/VsCode/HelloWorld/TransformerArt/train.py)

```python
def train():
    # 1. Model oluştur
    model = GPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
    model = model.to(device)
    
    # 2. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 3. Training loop
    for iter in range(max_iters):
        # Batch al
        xb, yb = data_loader.get_batch('train')
        
        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluate
        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"Step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")
```

### 🎯 Hiperparametreler

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `batch_size` | 32 | Paralel sequence sayısı |
| `block_size` | 128 | Context length |
| `max_iters` | 1000 | Training iterations |
| `learning_rate` | 3e-4 | AdamW learning rate |
| `eval_interval` | 100 | Evaluation frequency |

**Learning Rate Schedule:**
- Constant 3e-4 (basit)
- Alternatif: Warmup + Cosine decay

### 📊 Training Süreci

```
Step    0: train loss 6.2443, val loss 6.2459  ← Random başlangıç
Step  100: train loss 4.9251, val loss 4.9413  ← Öğrenmeye başladı
Step  200: train loss 4.3972, val loss 4.4524
Step  300: train loss 4.0376, val loss 4.1358
Step  400: train loss 3.8804, val loss 3.9958
Step  500: train loss 3.7768, val loss 3.9083
Step  600: train loss 3.7164, val loss 3.8530
Step  700: train loss 3.6581, val loss 3.8073
Step  800: train loss 3.6101, val loss 3.7897
Step  900: train loss 3.5336, val loss 3.7410
Step  999: train loss 3.4780, val loss 3.6632  ← Final
```

**Loss Reduction:** 6.24 → 3.48 (**~44% improvement**)

### 🔍 Ne Öğrendi?

Model şunları öğrendi:
- ✅ Türkçe karakter dizilimleri
- ✅ Kelime yapıları
- ✅ Bazı yaygın kelimeler
- ✅ Temel dilbilgisi kalıpları

---

## 9. Fine-Tuning

### 📖 Teori

**Fine-tuning**, pre-trained modeli specific task için uyarlar.

**Amaç:**
- Instruction-following öğren
- Kullanıcı talimatlarını takip et
- Tutarlı yanıtlar üret

**Dataset:** GPT-4-Self-Instruct-Turkish (1000 samples)

### 🧮 Format

**Instruction Format:**
```
<INST>instruction</INST><RESP>response</RESP>
```

**Örnek:**
```
<INST>Türkiye'nin başkenti neresidir?</INST>
<RESP>Türkiye'nin başkenti Ankara'dır.</RESP>
```

### 💻 Fine-Tuning Süreci

**Dosya:** [`Fine_Tune/fine_tune.py`](file:///c:/Users/w/Desktop/Kodlama/VsCode/HelloWorld/TransformerArt/Fine_Tune/fine_tune.py)

```python
def fine_tune():
    # 1. Instruction dataset yükle
    dataset = InstructionDataset(max_samples=1000)
    
    # 2. Pre-trained model yükle
    model, config, _ = load_model_hf_format(pretrained_model_path, device)
    
    # 3. Vocabulary genişlet (special tokens için)
    if dataset.vocab_size != config['vocab_size']:
        # Embedding layer'ı genişlet
        expand_embeddings(model, dataset.vocab_size)
    
    # 4. Optimizer (düşük learning rate!)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 5. Fine-tuning loop
    for iter in range(ft_max_iters):
        xb, yb = dataset.get_batch('train')
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 🎯 Fine-Tuning Hiperparametreleri

| Parametre | Pre-Training | Fine-Tuning | Neden? |
|-----------|--------------|-------------|--------|
| `learning_rate` | 3e-4 | 1e-4 | Küçük değişiklikler |
| `batch_size` | 32 | 8 | Daha az data |
| `max_iters` | 1000 | 500 | Hızlı converge |
| `block_size` | 128 | 256 | Uzun instructions |

**Neden düşük LR?**
- Pre-trained weights'i korumak
- Catastrophic forgetting'i önlemek
- Stable fine-tuning

### 📊 Beklenen Sonuçlar

```
Step    0: train loss 3.8421, val loss 3.8567  ← Pre-trained'den başla
Step   50: train loss 2.9156, val loss 2.9289  ← Instruction format öğreniyor
Step  100: train loss 2.5234, val loss 2.5456
Step  200: train loss 2.1567, val loss 2.2134
Step  300: train loss 1.9876, val loss 2.0543
Step  400: train loss 1.8765, val loss 1.9876
Step  499: train loss 1.8234, val loss 1.9456  ← Final
```

---

## 10. Önemli Parametreler

### 📊 Model Boyutu vs Performance

| n_embd | n_head | n_layer | Parameters | Training Time | Performance |
|--------|--------|---------|------------|---------------|-------------|
| 64 | 2 | 2 | ~200K | Çok hızlı | Düşük |
| 128 | 4 | 4 | ~936K | Hızlı | Orta |
| 256 | 8 | 6 | ~10M | Orta | İyi |
| 512 | 16 | 12 | ~100M | Yavaş | Çok iyi |

**Bizim Model:** 128/4/4 → 936K params (öğrenme için ideal)

### 🎯 Parametre Seçimi Rehberi

**1. Embedding Dimension (n_embd):**
- Küçük (64-128): Hızlı, basit tasks
- Orta (256-512): Genel kullanım
- Büyük (768-1024): Complex tasks

**2. Attention Heads (n_head):**
- n_embd % n_head == 0 olmalı!
- Daha fazla head → Daha fazla farklı ilişki
- Tipik: n_embd / n_head = 32-64

**3. Number of Layers (n_layer):**
- Az (2-4): Hızlı, basit
- Orta (6-12): Standart
- Çok (24+): GPT-3 seviyesi

**4. Context Length (block_size):**
- Kısa (128-256): Hızlı, az memory
- Uzun (512-2048): Daha fazla context
- Trade-off: Memory O(T²)

**5. Batch Size:**
- Küçük (8-16): Az memory, noisy gradients
- Büyük (32-64): Stable training
- GPU memory'e göre ayarla

**6. Learning Rate:**
- Pre-training: 1e-4 to 3e-4
- Fine-tuning: 1e-5 to 1e-4
- Adam/AdamW için tipik

---

## 11. Debugging ve İpuçları

### 🐛 Yaygın Hatalar

**1. CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```

**Çözüm:**
```python
# config.py'de küçült
batch_size = 16  # 32 yerine
block_size = 64  # 128 yerine
n_embd = 64      # 128 yerine
```

**2. Loss NaN**
```
Step 100: train loss nan
```

**Nedenler:**
- Learning rate çok yüksek
- Gradient explosion
- Numerical instability

**Çözüm:**
```python
# Learning rate küçült
learning_rate = 1e-4  # 3e-4 yerine

# Gradient clipping ekle
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**3. Token ID Out of Range**
```
RuntimeError: index out of range
```

**Çözüm:**
```python
# Token validation ekle
tokens = [min(t, vocab_size-1) for t in tokens]
```

**4. Loss Düşmüyor**

**Kontrol listesi:**
- ✅ Learning rate uygun mu?
- ✅ Data doğru mu yükleniyor?
- ✅ Model device'da mı?
- ✅ Optimizer doğru mu?

### 📈 Training İzleme

**1. Loss Curves:**
```python
import matplotlib.pyplot as plt

plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

**İdeal:**
- Train loss düşüyor
- Val loss düşüyor
- Gap küçük (overfitting yok)

**2. Gradient Norms:**
```python
total_norm = 0
for p in model.parameters():
    param_norm = p.grad.data.norm(2)
    total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5
print(f"Gradient norm: {total_norm}")
```

**İdeal:** 0.1 - 10 arası

**3. Text Generation:**
```python
# Her 100 iteration'da test et
if iter % 100 == 0:
    sample = model.generate(context, max_new_tokens=50)
    print(data_loader.decode(sample[0].tolist()))
```

### 💡 İpuçları

**1. Başlangıç:**
- ✅ Küçük model ile başla
- ✅ Az data ile test et
- ✅ Overfit edebiliyor mu kontrol et

**2. Scaling:**
- ✅ Önce model boyutunu artır
- ✅ Sonra data'yı artır
- ✅ Son olarak training time'ı artır

**3. Fine-Tuning:**
- ✅ Düşük learning rate kullan
- ✅ Az iteration yeterli
- ✅ Validation loss'u izle

**4. GPU Kullanımı:**
```python
# GPU memory kullanımını izle
print(torch.cuda.memory_allocated() / 1024**2, "MB")
print(torch.cuda.memory_reserved() / 1024**2, "MB")
```

---

## 🎓 Özet: Adım Adım Eğitim

### Pre-Training (Sıfırdan)

```bash
# 1. Sanal ortam
python -m venv venv
.\venv\Scripts\activate

# 2. Bağımlılıklar
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 3. BPE Tokenizer eğit + Model eğit
python train.py

# Beklenen süre: 2-3 dakika (CUDA)
# Beklenen loss: 6.24 → 3.48
```

### Fine-Tuning

```bash
# 1. Fine-tuning klasörüne git
cd Fine_Tune

# 2. Dataset kütüphaneleri
pip install datasets huggingface_hub

# 3. Fine-tuning
python fine_tune.py

# Beklenen süre: 1-2 dakika
# Beklenen loss: 3.84 → 1.82
```

### Inference

```bash
# Pre-trained model
python generate.py

# Fine-tuned model
cd Fine_Tune
python inference.py
```

---

## 📚 Kaynaklar

**Papers:**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer
- [Neural Machine Translation with BPE](https://arxiv.org/abs/1508.07909) - BPE Algorithm
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - GPT-3

**Code:**
- [Andrej Karpathy - nanoGPT](https://github.com/karpathy/nanoGPT)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)

**Tutorials:**
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

---

## 🎉 Sonuç

Bu rehberde:
- ✅ BPE tokenization'ı sıfırdan kodladık
- ✅ Self-attention mekanizmasını anladık
- ✅ Transformer bloklarını birleştirdik
- ✅ Pre-training yaptık
- ✅ Fine-tuning yaptık
- ✅ Her adımda matematik + kod gördük

**Sonraki Adımlar:**
1. Farklı hiperparametreler dene
2. Daha büyük dataset kullan
3. Model boyutunu artır
4. Farklı tasks için fine-tune et

**Başarılar!** 🚀
