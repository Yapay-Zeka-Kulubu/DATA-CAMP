# IMDB Duygu Analizi (Sentiment Analysis) Projesi

Bu proje, IMDB film yorumları veri kümesini kullanarak yorumların **Pozitif** veya **Negatif** olduğunu tahmin eden bir makine öğrenmesi modelidir. Proje; veri temizleme (NLP), metin sayısallaştırma (TF-IDF) ve sınıflandırma (Lojistik Regresyon) aşamalarını içerir.

## Proje Akışı


1.  **Veri Yükleme:** Hugging Face `datasets` kütüphanesi kullanılarak 50.000 yorumluk IMDB veri seti yüklenir.
2.  **Ön İşleme (Preprocessing):**
    * Metinler küçük harfe dönüştürülür.
    * Sayılar ve noktalama işaretleri temizlenir.
    * **Stopwords** (the, is, in vb.) çıkarılır.
    * **Lemmatization** ile kelimeler köklerine indirgenir.
    * Gereksiz boşluklar temizlenir.
3.  **Vektörleştirme (TF-IDF):** Metin verileri, kelime frekanslarına göre (n-gram: 1,2) sayısal matrislere dönüştürülür.
4.  **Model Eğitimi:** Lojistik Regresyon algoritması kullanılarak model eğitilir.
5.  **Değerlendirme:** Modelin başarısı Accuracy, Precision, Recall ve F1 skorları ile ölçülür ve Karmaşıklık Matrisi (Confusion Matrix) oluşturulur.

## Kullanılan Teknolojiler

* **Python 3.x**
* **Scikit-Learn:** Makine öğrenmesi ve metin işleme.
* **NLTK:** Doğal dil işleme (Lemmatization, Stopwords).
* **Matplotlib:** Görselleştirme.
* **Hugging Face Datasets:** Veri setine erişim.

## Dosya Yapısı

* `main.py`: Ana uygulama kodu.
* `results/`: Model çıktıları (Metrikler ve Confusion Matrix grafiği).
* `requirements.txt`: Gerekli kütüphaneler.

## Kurulum ve Çalıştırma

1. Kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt