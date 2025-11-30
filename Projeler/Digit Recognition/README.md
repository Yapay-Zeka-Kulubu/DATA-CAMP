# MNIST Veri Seti

## İçindekiler
1. [Giriş](#giriş)
2. [Veri Seti](#veri-seti)
3. [Kütüphaneler](#kütüphaneler)


## Giriş
MNIST (Modified National Institute of Standards and Technology) veri seti ile 0'dan 9'a kadar olan rakam görsellerini tanıyan bir Digit Recognition projesi. CNN ile kendi mimarinizi oluşturmanız daha sonra onu geliştirmeniz önerilir.

## Veri-Seti
- **Veri Tipi:** Görüntü
- **Görüntü Boyutu:** 28x28 piksel
- **Renk:** Gri tonlama (0-255 arası piksel değerleri)
- **Sınıf Sayısı:** 10 (0'dan 9'a kadar rakamlar)
- **Eğitim Verisi Sayısı:** 60,000 örnek
- **Test Verisi Sayısı:** 10,000 örnek

## Veri Setini İndirme
MNIST veri seti, Keras kütüphanesi içinde yerleşik olarak gelir ve aşağıdaki komutla kolayca indirilebilir:

```python
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```
### Kütüphaneler
- **NumPy:** Sayısal işlemler için
- **Matplotlib:** Görselleştirme için
- **Keras:** Derin öğrenme modelleri oluşturmak için
- **TensorFlow:** Keras arka ucunu sağlamak için (opsiyonel)
