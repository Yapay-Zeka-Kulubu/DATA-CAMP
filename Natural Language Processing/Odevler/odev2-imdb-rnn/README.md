# IMDb Sentiment Analysis  
## Klasik Makine Öğrenmesi vs Derin Öğrenme (LSTM)

Bu projede IMDb Sentiment Dataset kullanılarak duygu analizi gerçekleştirilmiştir.  
Amaç, **klasik makine öğrenmesi yaklaşımları** ile **derin öğrenme (RNN / LSTM)** tabanlı modellerin performanslarını karşılaştırmak ve aralarındaki farkları analiz etmektir.


## 📊 Kullanılan Veri Seti

- **IMDb Sentiment Dataset**
- 50.000 film yorumu
- Dengeli sınıf dağılımı (Pozitif / Negatif)
- Eğitim / test ayrımı hazır şekilde kullanılmıştır:
  - Train: 25.000
  - Test: 25.000

---

## 🧹 Metin Ön İşleme (Preprocessing)

### Ortak Adımlar
- Küçük harfe çevirme
- Noktalama işaretlerinin temizlenmesi

### Klasik ML (Ödev 1)
- Stopwords temizleme
- Lemmatization
- TF-IDF vektörleştirme

### Derin Öğrenme (Ödev 2)
- Stopwords temizleme (opsiyonel)
- Tokenization
- Padding / Truncation
- **TF-IDF kullanılmamıştır**

Derin öğrenmede preprocessing daha hafif tutulmuştur çünkü anlamsal temsili **Embedding katmanı** öğrenmektedir.

---

## 🧠 Ödev 1 – Klasik Makine Öğrenmesi (ML)

### Kullanılan Yöntemler
- TF-IDF Vectorizer
- Logistic Regression

### Özellik Çıkarımı
- `max_features`
- `ngram_range`
- `stop_words`

TF-IDF kelime sırasını dikkate almaz, metni sabit boyutlu bir vektör olarak temsil eder.

---

## 🤖 Ödev 2 – Derin Öğrenme (RNN / LSTM)

### Embedding Katmanı

Embedding layer, kelimeleri one-hot encoding yerine **anlamlı, yoğun (dense) vektörler** olarak temsil eder.

Kullanılan parametreler:

- `vocab_size`: Modelin tanıyacağı maksimum kelime sayısı
- `embedding_dim`: Her kelimenin temsil edildiği vektör boyutu
- `max_length`: Her metnin sabit sequence uzunluğu

---

### Model Mimarisi (LSTM)

LSTM modeli, RNN’lerin yaşadığı **vanishing gradient (unutma) problemini** gate mekanizmaları ile çözer.

Kullanılan mimari:

- Embedding Layer  
- LSTM Layer  
- Dropout Layer  
- Dense (Sigmoid) Output Layer  

Dropout, modelin eğitim sırasında aşırı öğrenmesini (overfitting) azaltmak için eklenmiştir.

---

## 🏋️ Model Eğitimi

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Epoch: 5
- Batch Size: 64
- Validation Split: %20

---

## 📈 Model Değerlendirme

Her iki model için aşağıdaki metrikler hesaplanmıştır:

- Accuracy
- Precision
- Recall
- F1-score

Ayrıca her model için bir **confusion matrix** görselleştirmesi oluşturulmuştur.

Sonuçlar `results/` klasörü altında saklanmaktadır.

---

## 🔍 Karşılaştırma Analizi  
### TF-IDF + ML vs LSTM

| Kriter | TF-IDF + Logistic Regression | LSTM |
|------|-----------------------------|------|
| Performans | Yüksek ve stabil | Yüksek fakat dalgalı |
| Eğitim Süresi | Çok hızlı | Daha yavaş |
| Overfitting | Düşük | Daha yatkın |
| Kelime Sırası | Dikkate alınmaz | Dikkate alınır |
| Yorumlanabilirlik | Yüksek | Düşük |
| Model Karmaşıklığı | Düşük | Yüksek |

---

## 📌 Genel Değerlendirme

Klasik makine öğrenmesi modelleri, daha az hesaplama maliyeti ve daha stabil sonuçlar sunarken;  
LSTM tabanlı modeller, kelime sırası ve bağlam bilgisini kullanarak daha zengin temsiller öğrenebilmektedir.

Ancak bu avantaj, daha uzun eğitim süresi ve overfitting riski ile birlikte gelmektedir.

Bu çalışma, **klasik ML ve derin öğrenme yaklaşımlarının güçlü ve zayıf yönlerini** açıkça ortaya koymaktadır.

---

## 🚀 Sonuç

Bu projede:

- Metin verisi üzerinde farklı modelleme yaklaşımları uygulanmış
- Performans karşılaştırması yapılmış
- Sequence modeling kavramı pratik olarak gösterilmiştir

Proje, duygu analizi problemleri için uygun model seçiminin önemini vurgulamaktadır.
