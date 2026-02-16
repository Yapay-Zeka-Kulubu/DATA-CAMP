# Image Classification with ResNet (ImageNet-based)

## 📌 Proje Tanımı
Bu projede, derin öğrenme tabanlı bir görüntü sınıflandırma sistemi geliştirilmiştir. Model mimarisi, Deep Residual Learning yaklaşımına dayanan ResNet mimarisidir.

Proje; veri setinin hazırlanması, veri ön işleme, model eğitimi, test süreci ve performans değerlendirmesi olmak üzere uçtan uca bir makine öğrenmesi pipeline’ı sunmaktadır.

Tüm kodlar modüler, yeniden kullanılabilir ve başka kullanıcıların modeli kolayca test edebilmesi amacıyla tasarlanmıştır.

## 📁 Klasör Yapısı
ImageNet/
├── README.md
├── config.py
├── dataset.py
├── model.py
├── train.py
├── eval.py
├── utils.py
└── main.py

## 📊 Veri Seti
- Veri Türü: Görüntü sınıflandırma
- Girdi: RGB görüntüler
- Çıkış: Çok sınıflı etiketler
- Veri Formatı: ImageFolder

Veri %70 eğitim, %15 doğrulama, %15 test olarak bölünür.

## ⚙️ Veri Ön İşleme
Eğitim verisi için yeniden boyutlandırma, veri artırma ve normalize işlemleri uygulanır.
Test ve doğrulama verilerinde yalnızca yeniden boyutlandırma ve normalize yapılır.

## 🧠 Model
ResNet-18 veya ResNet-50 mimarisi kullanılır. Son katman sınıf sayısına göre yeniden tanımlanır.

## 🏋️ Model Eğitimi
- Loss: CrossEntropyLoss
- Optimizer: Adam
- En iyi doğrulama başarımı gösteren model kaydedilir.

## 💾 Model Ağırlıkları
Eğitim sonunda en iyi model:
weights/best_model.pth

## 📈 Test ve Değerlendirme
Test verisi üzerinde aşağıdaki metrikler hesaplanır:
- Accuracy
- Precision (Macro)
- Recall (Macro)
- F1-Score (Macro)

## ▶️ Çalıştırma
Eğitim:
python main.py --mode train

Test:
python main.py --mode test --weights weights/best_model.pth

## 👥 Başka Kullanıcılar İçin
1. Repo klonlanır
2. requirements.txt kurulur
3. Ağırlık dosyası indirilir
4. Test komutu çalıştırılır

## 🎯 Sonuç
Bu proje akademik ve endüstriyel standartlara uygun, modüler ve yeniden üretilebilir bir görüntü sınıflandırma sistemidir.
