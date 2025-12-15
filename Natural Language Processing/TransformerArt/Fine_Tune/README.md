# Instruction Fine-Tuning

Bu klasör, pre-trained GPT modelini instruction-following için fine-tune etmeyi içerir.

## 📁 Dosyalar

- `ft_config.py` - Fine-tuning konfigürasyonu
- `dataset_loader.py` - HuggingFace dataset yükleyici
- `fine_tune.py` - Fine-tuning scripti
- `inference.py` - İnteraktif inference

## 🎯 Amaç

Pre-trained GPT modelini instruction-response formatında fine-tune ederek, modelin kullanıcı talimatlarını takip etmesini sağlamak.

## 📊 Dataset

**HuggingFace:** `CausalLM/GPT-4-Self-Instruct-Turkish`
- Türkçe instruction-response çiftleri
- GPT-4 tarafından oluşturulmuş
- 1000 sample kullanılıyor (demo için)

## 🚀 Kullanım

### 1. Bağımlılıkları Kur

```bash
pip install datasets huggingface_hub
```

### 2. Fine-Tuning Yap

```bash
cd Fine_Tune
python fine_tune.py
```

**Beklenen Çıktı:**
```
============================================================
INSTRUCTION FINE-TUNING
============================================================

Loading Instruction Dataset
============================================================
Downloading from HuggingFace: CausalLM/GPT-4-Self-Instruct-Turkish
Loaded 1000 samples

Loading pre-trained BPE tokenizer...
Vocabulary size (with special tokens): 504

Formatting instruction-response pairs...
Processed 100/1000 samples
...

📊 Dataset Statistics:
   Total tokens: 150,000
   Train tokens: 135,000
   Validation tokens: 15,000
============================================================

Loading pre-trained model...
Model parameters: 940,000

============================================================
Starting fine-tuning...
============================================================
Step    0: train loss 3.8421, val loss 3.8567
Step   50: train loss 2.9156, val loss 2.9289
Step  100: train loss 2.5234, val loss 2.5456
...
Step  499: train loss 1.8234, val loss 1.9456

============================================================
Fine-tuning completed!
============================================================

✅ Fine-tuned model saved to: fine_tuned_model.pt
```

### 3. İnteraktif Inference

```bash
python inference.py
```

**Kullanım:**
```
📝 Instruction: Türkiye'nin başkenti neresidir?
   Max tokens (default 200): 100
   Temperature (default 0.7): 0.7

🤖 Generating response...

------------------------------------------------------------
Türkiye'nin başkenti Ankara'dır. 1923 yılında Cumhuriyet'in 
ilanından sonra başkent olarak seçilmiştir.
------------------------------------------------------------
```

## ⚙️ Konfigürasyon

```python
# ft_config.py
ft_batch_size = 8          # Küçük batch size
ft_block_size = 256        # Uzun context
ft_max_iters = 500         # Az iteration
ft_learning_rate = 1e-4    # Düşük learning rate
max_samples = 1000         # 1000 sample
```

## 📝 Format

**Instruction Format:**
```
<INST>instruction</INST><RESP>response</RESP>
```

**Örnek:**
```
<INST>Python'da liste nasıl oluşturulur?</INST>
<RESP>Python'da liste köşeli parantez kullanılarak oluşturulur: 
my_list = [1, 2, 3, 4, 5]</RESP>
```

## 🎓 Öğrenilen Kavramlar

- **Transfer Learning:** Pre-trained model kullanma
- **Fine-Tuning:** Specific task için model adaptasyonu
- **Instruction Following:** Talimat takip etme
- **Special Tokens:** Format için özel tokenler
- **Lower Learning Rate:** Fine-tuning için düşük LR

## 📊 Beklenen Sonuçlar

- Initial loss: ~3.8
- Final loss: ~1.8-2.0
- Model instruction formatını öğrenir
- Türkçe talimatları takip eder
- Tutarlı yanıtlar üretir
