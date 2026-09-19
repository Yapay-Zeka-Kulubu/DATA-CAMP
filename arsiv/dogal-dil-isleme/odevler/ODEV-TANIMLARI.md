# 📘 ÖDEV 1 README – IMDb Sentiment Analysis (Duygu Analizi Projesi)

## 🎯 Ödevin Konusu

Bu ödevde **IMDb film yorumları veri seti** kullanılarak bir **Duygu Analizi (Sentiment Analysis)** modeli geliştirilmesi amaçlanmaktadır.

**Amaç:**
Bir film yorumunun **olumlu (positive)** veya **olumsuz (negative)** olduğunu, **makine öğrenmesi yöntemleri** kullanarak sınıflandırmaktır.

---

## 📌 Ödevin Hedefleri

Bu ödev ile aşağıdaki kazanımların elde edilmesi hedeflenmektedir:

* Doğal Dil İşleme (NLP) temel kavramlarını uygulamak
* Metin ön işleme (preprocessing) adımlarını gerçekleştirmek
* TF-IDF yöntemi ile özellik çıkarımı yapmak
* Basit bir makine öğrenmesi modeli eğitmek
* Model performansını değerlendirmek
* Doğru ve düzenli bir proje yapısı oluşturmak

---

## 📂 Kullanılacak Veri Seti

### 🟦 IMDb Sentiment Dataset

* Toplam **50.000** film yorumu
* **Pozitif / Negatif** duygu etiketleri
* HuggingFace üzerinden indirilmektedir

Veri seti aşağıdaki kod ile yüklenebilir:

```python
from datasets import load_dataset
dataset = load_dataset("imdb")
```

---

## 🧭 Ödevde Yapılması Gerekenler (Zorunlu Adımlar)

### ✔ 1) Veri Setini Yükleme

* IMDb dataset’i HuggingFace üzerinden indirilecektir
* Eğitim (train) ve test (test) ayrımı kullanılacaktır

---

### ✔ 2) Metin Ön İşleme (Preprocessing)

Aşağıdaki adımların **tamamı** uygulanmalıdır:

* Metinleri küçük harfe çevirme
* Noktalama işaretlerini temizleme
* Sayıları kaldırma (opsiyonel)
* Stopwords temizleme
* Lemmatization veya stemming
* Gereksiz boşlukları silme

📌 **Not:** README dosyasında kullanılan preprocessing adımları ayrıca açıklanmalıdır.

---

### ✔ 3) Özellik Çıkarımı (TF-IDF)

Bu ödevde **TF-IDF kullanımı zorunludur**.

Beklenenler:

* TF-IDF vektörleştirici kullanılması
* En az **3 parametrenin** açıklanması

Örnek parametreler:

* `max_features`
* `ngram_range`
* `stop_words`

---

### ✔ 4) Makine Öğrenmesi Modeli Eğitimi

Aşağıdaki modellerden **bir tanesi** seçilmelidir:

* Logistic Regression (**önerilir**)
* Linear SVM
* Multinomial Naive Bayes

📌 **Not:** Seçilen modelin neden tercih edildiği README dosyasında açıklanmalıdır.

---

### ✔ 5) Model Değerlendirme

Aşağıdaki metrikler **zorunludur**:

* Accuracy
* Precision
* Recall
* F1-score

Ek olarak:

* Bir **confusion matrix** görselleştirmesi (grafik veya tablo) eklenmelidir

---

### ✔ 6) Kendi Cümleleriyle Test Yapma

Model, en az **5 farklı örnek cümle** ile test edilmelidir.

Örnek:

```
“This movie was boring and slow.” → Negative
```

---

### ✔ 7) Proje Yapısı

Proje aşağıdaki formatta teslim edilmelidir:

```text
project/
├── README.md
├── requirements.txt
├── sentiment_analysis.py
└── results/
    ├── metrics.txt
    └── confusion_matrix.png
```

---

## 📝 Beklenen Çıktılar

README dosyasında **mutlaka** yer almalıdır:

* Ödevin kısa açıklaması
* Uygulanan preprocessing adımları
* TF-IDF parametreleri
* Kullanılan model ve neden seçildiği
* Sonuç metrikleri
* Confusion matrix görseli
* Örnek tahminler

---

## 📦 Teslim Gereksinimleri

Teslim edilecekler:

* GitHub repository linki
* Tüm dosyaların eksiksiz olması
* Kodun çalışır durumda olması
* Doğru hazırlanmış `requirements.txt` dosyası


# 🟩 ÖDEV 2 – IMDb Sentiment Analysis (RNN / LSTM ile Derin Öğrenme)

## 🎯 Ödevin Konusu

Bu ödevde, **Ödev 1’de kullanılan IMDb veri seti**, bu kez **Derin Öğrenme tabanlı modeller** ile ele alınmıştır. Amaç, klasik makine öğrenmesi yaklaşımları ile **RNN / LSTM tabanlı modellerin performanslarını karşılaştırmak** ve aralarındaki farkları analiz etmektir.

---

## 📌 Ödevin Hedefleri

* Sıralı veri (sequence) mantığını anlamak
* RNN ve LSTM mimarilerinin çalışma prensibini öğrenmek
* Metin verisi üzerinde embedding kullanımını kavramak
* Klasik ML ve DL modellerinin karşılaştırmasını yapmak

---

## 🧭 Ödevde Yapılması Gerekenler (Zorunlu Adımlar)

✔ 1) Veri Seti

* IMDb Sentiment Dataset kullanılacaktır
* Eğitim / test ayrımı korunacaktır

✔ 2) Metin Ön İşleme

* Küçük harfe çevirme
* Noktalama işaretlerini temizleme
* Stopwords temizleme (opsiyonel)
* Tokenization
* Padding / Truncation

📌 Not: TF-IDF **kullanılmayacaktır**

---

✔ 3) Embedding Katmanı

* Embedding layer kullanılmalıdır
* `vocab_size`, `embedding_dim`, `max_length` parametreleri açıklanmalıdır

---

✔ 4) Model Mimarisi

Aşağıdaki modellerden **en az biri** kullanılmalıdır:

* Simple RNN
* LSTM (önerilir)

Model mimarisi README içinde açıklanmalıdır.

---

✔ 5) Model Eğitimi ve Değerlendirme

Zorunlu metrikler:

* Accuracy
* Precision
* Recall
* F1-score

Ek olarak:

* Confusion matrix görselleştirmesi

---

✔ 6) Karşılaştırma Analizi

README içinde aşağıdaki karşılaştırma yapılmalıdır:

* TF-IDF + ML (Ödev 1)
* RNN / LSTM (Ödev 2)

Karşılaştırma kriterleri:

* Performans
* Eğitim süresi
* Overfitting eğilimi
* Yorumlanabilirlik

---

✔ 7) Proje Yapısı

```
project/
├── README.md
├── requirements.txt
├── sentiment_analysis_ml.py
├── sentiment_analysis_rnn.py
└── results/
    ├── ml_metrics.txt
    ├── rnn_metrics.txt
    └── confusion_matrices/
```

---

# 🟨 ÖDEV 3 – LLM API Kullanarak Basit RAG Mimarisi

## 🎯 Ödevin Konusu

Bu ödevde, bir **Large Language Model (LLM)** API üzerinden kullanılarak, **Retrieval-Augmented Generation (RAG)** mimarisinin temel bir versiyonu oluşturulmuştur.

Amaç: Harici bir dokümana dayalı olarak, modelin **kontrollü ve bağlama bağlı cevap üretmesini** sağlamaktır.

---

## 📌 Ödevin Hedefleri

* LLM API kullanımı (OpenAI / Groq vb.)
* Prompt engineering temel prensipleri
* RAG mimarisinin mantığını kavramak
* Hallucination problemini azaltma

---

## 🧠 Kullanılan RAG Yaklaşımı

Bu ödevde **arayüz kullanılmamıştır**. Tüm işlemler **kod üzerinden** yapılmaktadır.

### 🔍 Retrieval (Bilgi Getirme)

* Dokümanlar dosya sisteminden yüklenir
* Metin paragraflara bölünür
* Kullanıcı sorusuna göre anahtar kelime eşleşmesi yapılır
* En alakalı bölümler seçilir

📌 Vector Database **kullanılmamıştır** (basic RAG)

---

### 🧩 Augmentation

Seçilen bağlam, system prompt içine eklenir:

* "Aşağıdaki dökümana dayanarak cevap ver"

Bu sayede model:

* Kaynak dışına çıkmaz
* Daha güvenilir cevaplar üretir

---

### ✨ Generation

* LLM API çağrısı yapılır
* Cevap sadece verilen bağlama dayanır

---

## 🧭 Ödevde Yapılması Gerekenler

✔ LLM API entegrasyonu

✔ Dosyadan doküman okuma

✔ Basit retrieval algoritması

✔ Prompt + context oluşturma

✔ En az 3 farklı soru ile test

---

## 📂 Proje Yapısı

```
project/
├── README.md
├── requirements.txt
├── rag_chat.py
└── data/
    └── document.txt
```

---

## 📌 Genel Değerlendirme

Bu üç ödev birlikte:

* Klasik NLP
* Derin Öğrenme
* Modern LLM tabanlı sistemler

arasındaki farkları **uygulamalı olarak** göstermektedir.

Bu yapı, NLP alanında uçtan uca bir öğrenme süreci sunmaktadır.


