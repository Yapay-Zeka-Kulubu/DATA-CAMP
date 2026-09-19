# Katkı ve Ödev Teslim Rehberi

Bu repo bir **eğitim reposudur**. Düzenin korunması, hem sizin hem de sizden sonra gelen katılımcıların işini kolaylaştırır.

---

## 📥 Ödev Teslimi

### 1. Repoyu fork'layın ve klonlayın

```bash
git clone https://github.com/<kullanici-adiniz>/DATA-CAMP.git
cd DATA-CAMP
git checkout -b odev/<modul>-<odev-adi>
```

Örnek: `git checkout -b odev/nlp-imdb-duygu-analizi`

### 2. Ödevinizi doğru klasöre koyun

İlgili modülün `odevler/` klasörü altında **kendi adınızla** bir klasör açın:

```
03-dogal-dil-isleme-atolyesi/odevler/odev1-imdb-duygu-analizi/
└── ad-soyad/
    ├── README.md
    ├── requirements.txt
    ├── <kodunuz>.py
    └── results/
        ├── metrics.txt
        └── confusion_matrix.png
```

### 3. README.md yazın

Her ödev teslimi bir `README.md` içermelidir:

- Ödevin kısa açıklaması
- Uygulanan adımlar (ön işleme, özellik çıkarımı, model)
- Kullanılan model ve **neden seçildiği**
- Sonuç metrikleri (accuracy, precision, recall, F1)
- Varsa görseller (confusion matrix vb.)
- Kodun nasıl çalıştırılacağı

### 4. Pull Request açın

PR başlığı: `[Modül] Ödev adı — Ad Soyad`
Örnek: `[NLP] Ödev 1 IMDb Duygu Analizi — Ayşe Yılmaz`

---

## ✍️ Commit Mesajları

Anlamlı ve Türkçe yazın. `Create file.py`, `Add files via upload`, `set` gibi mesajlardan kaçının.

```
✅ NLP ödev 1: TF-IDF + Logistic Regression modeli eklendi
✅ CV hafta 3 sunumu yüklendi
✅ RAG ödevi README'si güncellendi

❌ update
❌ Add files via upload
❌ Create main.py
```

---

## 📛 İsimlendirme Kuralları

| Kural | ✅ | ❌ |
|:--|:--|:--|
| Klasör/dosya adlarında **boşluk yok** | `atolye-projesi/` | `Atolye Projesi/` |
| **Türkçe karakter yok** | `dogal-dil-isleme/` | `doğal-dil-işleme/` |
| Küçük harf + tire (kebab-case) | `imdb-duygu-analizi.py` | `IMDB_Duygu Analizi.py` |
| README dosyaları büyük harfle | `README.md` | `readme.md` |

> Boşluk ve Türkçe karakter içeren yollar, farklı işletim sistemlerinde ve scriptlerde sorun çıkarır.

---

## 🔐 Güvenlik — API Anahtarları

**API anahtarlarını asla koda yazmayın ve commit etmeyin.**

Repo public olduğu için commit edilen bir anahtar, siz silseniz bile **git geçmişinde kalır** ve sızmış kabul edilir.

```python
# ❌ ASLA
client = Groq(api_key="gsk_xxxxxxxxxxxxxxxx")

# ✅ DOĞRU
import os
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
```

Anahtarlarınızı `.env` dosyasına koyun (`.gitignore` içindedir):

```bash
cp .env.example .env
```

---

## 🚫 Commit Etmeyin

- Sanal ortamlar (`venv/`, `myenv/`, `env/`)
- Model ağırlıkları (`*.pt`, `*.pth`, `*.bin`, `*.h5`) — boyutu şişirir
- Veri setleri — bunun yerine README'de indirme linki verin
- `__pycache__/`, `.ipynb_checkpoints/`
- `.env` ve her türlü gizli anahtar
- Editör/IDE ayarları (`.vscode/`, `.idea/`, `.github/copilot-instructions.md`)
- Aynı dosyanın kopyaları (`main-v2.py`, `deneme.py`, `xxx-main/`)
