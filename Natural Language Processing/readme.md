📘 ÖDEV 1 README – IMDb Sentiment Analysis (Duygu Analizi Projesi)
🎯 Ödevin Konusu

IMDb film yorumları veri setini kullanarak bir Duygu Analizi (Sentiment Analysis) modeli geliştirmeleri istenmektedir.
Amaç, bir film yorumunun olumlu (positive) veya olumsuz (negative) olduğunu makine öğrenmesi yöntemleriyle sınıflandırmaktır.

📌 Ödevin Hedefleri

Bu ödev ile :

Doğal Dil İşleme (NLP) temel kavramlarını uygulaması

Metin ön işleme (preprocessing) adımlarını öğrenmesi

TF-IDF ile özellik çıkarımı yapması

Basit bir makine öğrenmesi modelini eğitmesi

Model performansını doğru değerlendirebilmesi

Proje yapısı oluşturmayı öğrenmesi

beklenmektedir.

📂 Kullanılacak Veri Seti
🟦 IMDb Sentiment Dataset

50.000 film yorumu

Pozitif / negatif duygu etiketi

HuggingFace üzerinden indirilecektir

Veri setinin yüklenmesi için :

from datasets import load_dataset
dataset = load_dataset("imdb")


komutunu kullanacaktır.

🧭 Ödevde Yapılması Gerekenler (Zorunlu Adımlar)
✔ 1) Veri Setini Yükleme

IMDb dataset’i HuggingFace üzerinden indirilecek.

Eğitim ve test ayırımı doğru şekilde yapılacak.

✔ 2) Metin Ön İşleme (Preprocessing)

Aşağıdaki adımların hepsini uygulamalıdır:

Metinleri küçük harfe çevirme

Noktalama işaretlerini kaldırma

Sayıları kaldırma (opsiyonel)

Stopwords temizleme

Gerekiyorsa lemmatization / stemming

Gereksiz boşlukları silme

README içinde kendi preprocessing şemalarını açıklamaları zorunludur.

✔ 3) Özellik Çıkarımı

Bu projede TF-IDF kullanmak zorundadır.

Beklenen:

TF-IDF vectorizer kullanılması

En az 3 parametrenin açıklanması

Örneğin: max_features, ngram_range, stop_words

✔ 4) Makine Öğrenmesi Modeli Eğitimi

Şunlardan birini seçip kullanmalıdır:

Logistic Regression (önerilir)

Linear SVM

Multinomial Naive Bayes

Model seçimi ve gerekçesi README’de açıklanmalıdır.

✔ 5) Model Değerlendirme

Aşağıdaki metrikler zorunludur:

Accuracy

Precision

Recall

F1-score

Ayrıca :

Bir confusion matrix görselleştirmesi (grafik ya da tablo) eklemelidir.

✔ 6) Kendi Cümleleriyle Test

5 farklı örnek cümle yazıp model sonuçlarını göstermelidir.

Örnek:

“This movie was boring and slow.” → Negative

✔ 7) Proje Yapısı

Projeyi aşağıdaki formatta teslim etmelidir:

project/
│── README.md
│── requirements.txt
│── sentiment_analysis.py
│── results/
│     ├── metrics.txt
│     ├── confusion_matrix.png

📝 Beklenen Çıktılar

README dosyasında mutlaka bulunmalıdır:

Ödevin kısa açıklaması

Uygulanan preprocessing adımlarının listesi

TF-IDF parametreleri

Kullanılan model ve neden seçildiği

Sonuç metrikleri

Confusion matrix görseli

Örnek tahminler

📦 Teslim Gereksinimleri

Projeyi şu şekilde teslim etmelidir:

GitHub repo linki

Projede tüm dosyalar eksiksiz bulunmalıdır

Kod çalışır durumda olmalıdır

requirements.txt doğru olmalıdır

