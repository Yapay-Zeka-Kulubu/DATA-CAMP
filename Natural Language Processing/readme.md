📘 ÖDEV 1 README – IMDb Sentiment Analysis (Duygu Analizi Projesi)
🎯 Ödevin Konusu

Bu ödevde IMDb film yorumları veri seti kullanılarak bir Duygu Analizi (Sentiment Analysis) modeli geliştirilmesi beklenmektedir.
Amaç:
Bir film yorumunun olumlu (positive) veya olumsuz (negative) olduğunu makine öğrenmesi yöntemleri ile sınıflandırmaktır.

📌 Ödevin Hedefleri

Bu ödev ile aşağıdaki kazanımlar elde edilmelidir:

Doğal Dil İşleme (NLP) temel kavramlarını uygulamak

Metin ön işleme (preprocessing) adımlarını gerçekleştirmek

TF-IDF ile özellik çıkarımı yapmak

Basit bir makine öğrenmesi modeli eğitmek

Model performansını değerlendirmek

Doğru ve düzenli bir proje yapısı oluşturmak

📂 Kullanılacak Veri Seti
🟦 IMDb Sentiment Dataset

50.000 film yorumu

Pozitif / negatif duygu etiketi

HuggingFace üzerinden indirilecektir

Veri seti şu komutla yüklenebilir:
```python
from datasets import load_dataset
dataset = load_dataset("imdb")
```


🧭 Ödevde Yapılması Gerekenler (Zorunlu Adımlar)
✔ 1) Veri Setini Yükleme

IMDb dataset’i HuggingFace üzerinden indirilecek

Eğitim / test ayrımı kullanılacak

✔ 2) Metin Ön İşleme (Preprocessing)

Aşağıdaki adımların tamamı uygulanmalıdır:

Metinleri küçük harfe çevirme

Noktalama işaretlerini temizleme

Sayıları kaldırma (opsiyonel)

Stopwords temizleme

Lemmatization veya stemming

Gereksiz boşlukları silme

README içinde kullanılan preprocessing adımları ayrıca açıklanmalıdır.

✔ 3) Özellik Çıkarımı (TF-IDF)

Bu ödevde TF-IDF kullanmak zorunludur.

Beklenenler:

TF-IDF vektörleştirici kullanılması

En az 3 parametrenin açıklanması

Örnek: max_features, ngram_range, stop_words

✔ 4) Makine Öğrenmesi Modeli Eğitimi

Aşağıdaki modellerden biri seçilmelidir:

Logistic Regression (önerilir)

Linear SVM

Multinomial Naive Bayes

Model seçiminin gerekçesi README’de açıklanmalıdır.

✔ 5) Model Değerlendirme

Aşağıdaki metrikler zorunludur:

Accuracy

Precision

Recall

F1-score

Ek olarak:

Bir confusion matrix görselleştirmesi (grafik veya tablo) eklenmelidir.

✔ 6) Kendi Cümleleriyle Test Yapma

En az 5 farklı örnek cümle test edilmelidir.

Örnek:

“This movie was boring and slow.” → Negative

✔ 7) Proje Yapısı

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
📝 Beklenen Çıktılar

README dosyasında mutlaka yer almalıdır:

Ödevin kısa açıklaması

Uygulanan preprocessing adımları

TF-IDF parametreleri

Kullanılan model ve neden seçildiği

Sonuç metrikleri

Confusion matrix görseli

Örnek tahminler

📦 Teslim Gereksinimleri

Teslim edilecekler:

GitHub repo linki

Tüm dosyaların eksiksiz olması

Kodun çalışır durumda olması

Doğru hazırlanmış requirements.txt

