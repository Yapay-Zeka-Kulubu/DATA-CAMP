# DATA KAMP — 4. Sezon: Yapay Zekâya Giriş Eğitimi

**T.C. Erciyes Üniversitesi Yapay Zekâ Kulübü**

Bu repo, Erciyes Üniversitesi Yapay Zekâ Kulübü tarafından düzenlenen **DATA KAMP** eğitiminin ders materyallerini, ödevlerini ve projelerini barındırır.

| | |
|:--|:--|
| 📅 **Başlangıç** | 17.09.2026 |
| 📍 **Yer** | ERÜ Mühendislik Fakültesi |
| 🌐 **Web** | https://erciyesyapayzeka.com.tr/ |

---

## 🗺️ Eğitim Yapısı

Eğitim **ortak bir giriş bölümüyle** başlar, ardından katılımcılar iki atölyeden birini seçer.
Projeler eğitimle **paralel** ilerler — her modülün kendi ödev ve proje klasörü vardır.

```
                  ┌─────────────────────────────┐
                  │  01 — Yapay Zekâya Giriş    │
                  │      (ortak bölüm)          │
                  └──────────────┬──────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
   ┌───────────────────────────┐   ┌───────────────────────────┐
   │ 02 — Bilgisayarlı Görü    │   │ 03 — Doğal Dil İşleme     │
   │        Atölyesi           │   │        Atölyesi           │
   └───────────────────────────┘   └───────────────────────────┘
```

---

## 📚 Modüller

### [01 — Yapay Zekâya Giriş](01-yapay-zekaya-giris/) · *ortak bölüm*

| Hafta | Konu |
|:--|:--|
| 1 | Yapay Zekâ Nedir? Ne Değildir? |
| 2 | Veri ve Bilgi Analizi |
| 3 | Makine Öğrenmesi Temelleri |
| 4 | Model Tasarımı, Seçimi ve Değerlendirme |
| 5 | Yapay Sinir Ağlarına Giriş |
| 6 | Derin Modeller ve Konfigürasyon |

### [02 — Bilgisayarlı Görü Atölyesi](02-bilgisayarli-goru-atolyesi/)

| Hafta | Konu |
|:--|:--|
| 1 | Görüntü Oluşturma ve İşleme Temelleri |
| 2 | Görüntü Sınıflandırma |
| 3 | CNN Temelli Modeller ve Transfer Learning |
| 4 | Nesne Tespiti ve Değerlendirme |
| — | Atölye Projesi |

### [03 — Doğal Dil İşleme Atölyesi](03-dogal-dil-isleme-atolyesi/)

| Hafta | Konu |
|:--|:--|
| 1 | NLP'ye Giriş ve Dilin Temsili |
| 2 | Word Embeddings |
| 3 | Klasik Modeller |
| 4 | Derin Öğrenme ile NLP |
| 5 | Transformer, Fine-Tuning, RAG ve Agentler |
| — | Atölye Projesi |

---

## 📁 Repo Düzeni

```
DATA-CAMP/
├── 01-yapay-zekaya-giris/        # ortak bölüm: haftalar + ödevler + projeler
├── 02-bilgisayarli-goru-atolyesi/
├── 03-dogal-dil-isleme-atolyesi/
└── arsiv/                        # geçmiş sezonların materyalleri
    ├── makine-ogrenmesi/
    ├── bilgisayarli-goru/
    └── dogal-dil-isleme/
```

Her modül aynı düzeni kullanır: haftalık klasörler + `odevler/` + proje klasörü.
Her hafta klasöründe o haftanın sunumu, kodu ve notları bulunur.

> 📦 Önceki sezonlarda işlenen dersler, ödevler ve projeler [`arsiv/`](arsiv/) klasöründe korunmaktadır.

---

## 🚀 Başlarken

```bash
git clone https://github.com/Yapay-Zeka-Kulubu/DATA-CAMP.git
cd DATA-CAMP

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

Her ödev ve projenin kendi `requirements.txt` dosyası vardır:

```bash
pip install -r <ödev-klasörü>/requirements.txt
```

**API anahtarları** koda yazılmaz. `.env.example` dosyasını `.env` olarak kopyalayıp kendi anahtarlarınızı girin:

```bash
cp .env.example .env
```

---

## 🤝 Katkı

Ödev teslimi, branch ve commit kuralları için: [CONTRIBUTING.md](CONTRIBUTING.md)

## 👨‍🏫 Eğitmenler

- [Muhammet Özdemir](https://github.com/mr-ozdemir)
- [Kadir Yönak](https://github.com/kadiryonak)

## 🔗 Ek Kaynaklar

- [Proje örnekleri — best-of-ml-python](https://github.com/ml-tooling/best-of-ml-python)
- [Microsoft — AI for Beginners](https://github.com/microsoft/AI-For-Beginners)
- [Orta–ileri seviye eğitim](https://carpedm30.notion.site/AI-Compiler-Study-2cc71f48eb1140d09a439ab0b10bdb7b)
