# Metot Geliştirme Ödevi

## Tablo A: Makale Kimliği ve Kapsam

| Alan | Bilgi |
| :--- | :--- |
| **Makale başlığı** | Plant leaf disease classification using EfficientNet deep learning model |
| **Yazarlar** | Ümit Atila, Murat Uçar, K. Akyol, E. Uçar |
| **Yıl** | 2021 |
| **Yayın türü** | Dergi (Ecological Informatics) |
| **İndeks bilgisi** | SCI-E (Q1) |
| **Problem tanımı** | Tarımsal üretimde büyük kayıplara yol açan bitki hastalıklarının, mobil cihazlarda çalışabilecek hafif ve yüksek başarılı modellerle tespiti. |
| **Temel katkı** | EfficientNet mimarisinin bitki hastalıklarında kullanımı araştırılmış, AlexNet gibi eski modellere göre çok daha az parametre ile %99.91 başarı elde edilmiştir. |
| **Kod veya repo** | Var (GitHub üzerinde açık kaynak EfficientNet implementasyonları) |

## Tablo B: Veri Setleri ve Protokol (Makale bazlı)

| Veri seti | Örnek sayısı | Sınıf sayısı | Bölme (train/val/test) |
| :--- | :--- | :--- | :--- |
| PlantVillage | 55,448 | 38 | %80 Train / %10 Val / %10 Test |

## Tablo C: Veri Ön İşleme ve Artırma (Makale bazlı)

| Adım | Uygulama | Parametreler | Amaç | Not |
| :--- | :--- | :--- | :--- | :--- |
| **Normalizasyon** | Rescaling | [0, 1] | Piksel değerlerini normalize etmek. | - |
| **Yeniden boyutlama** | Resize | 224x224 (B0) | Model girişine uygun hale getirmek. | EfficientNet varyasyonuna göre değişir. |
| **Augmentation** | Rotation, Flip | - | Veri çeşitliliğini artırmak. | - |
| **Etiket işleme** | One-Hot | - | Kategorik sınıflandırma için. | - |

## Tablo D: Model Mimarisi (Makale bazlı)

| Bileşen | Seçim | Detay | Gerekçe |
| :--- | :--- | :--- | :--- |
| **Omurga (backbone)** | EfficientNet | B0 - B7 | Yüksek parametre verimliliği ve başarı. |
| **Başlık (head)** | Dense Layer | Global Avg Pool -> Softmax | Sınıflandırma katmanı. |
| **Aktivasyonlar** | Swish | x * sigmoid(x) | ReLU'ya göre daha iyi performans (Google önerisi). |
| **Normalizasyon** | Batch Norm | - | Eğitim kararlılığı. |
| **Kayıp fonksiyonu** | Categorical Crossentropy | - | Çok sınıflı problem standardı. |

## Tablo E: Eğitim Parametreleri (Makale bazlı)

| Parametre | Değer |
| :--- | :--- |
| **Optimizer** | Adam |
| **Öğrenme oranı** | 0.0001 (Başlangıç 0.001) |
| **LR scheduler** | Step Decay |
| **Batch size** | 32 |
| **Epoch** | 50 |
| **Weight decay** | Belirtilmemiş |
| **Erken durdurma** | Var (Validation loss takibi) |
| **Donanım** | NVIDIA GPU |
| **Seed ve determinism** | Belirtilmemiş |

## Tablo F: Sonuçlar ve Karşılaştırmalar (Makale bazlı)

| Veri seti | Metrikler | Sonuç | Baz çizgi | İyileştirme | Not |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PlantVillage | Accuracy | %99.91 | AlexNet (%99.35) | Daha az parametre ile daha yüksek başarı. | B5 modeli en iyi sonucu vermiştir. |

---

# Kendi Önerdiğiniz Metot İçin Zorunlu Tablolar

## Tablo H: Kullanılacak Veri Setleri Matrisi

| Veri seti | Kullanım amacı | Dahil mi | Dahil edilme gerekçesi | Risk / kısıt |
| :--- | :--- | :--- | :--- | :--- |
| **Proje veri seti 1** (PlantVillage) | Train / Val | Evet | Literatür standardı ve temel eğitim verisi. | Laboratuvar ortamı verisi olduğu için saha testinde yanıltıcı olabilir. |
| **Ek veri seti** (Real World Images) | Test / External | Evet | Modelin gerçek tarla koşullarındaki performansını (gürültü dayanıklılığını) ölçmek için. | Etiketleme doğruluğu riski olabilir. |

## Tablo I: Önerilen Uçtan Uca Pipeline

| Aşama | Girdi | Çıktı | Yöntem | Parametreler |
| :--- | :--- | :--- | :--- | :--- |
| **Veri alma** | Ham Görüntüler | Klasör Yapısı | PlantVillage ve Dış Veri birleştirme | - |
| **Temizleme** | Ham Veri | Temiz Veri | Bulanık/Hatalı görsel eleme | - |
| **Ön işleme** | Görüntü | Tensor | AutoAugment ve Normalizasyon | - |
| **Eğitim** | Tensor | Ağırlıklar | Transfer Learning (Fine-tuning) | Epoch: 30, Batch: 32 |
| **Değerlendirme** | Tahminler | Metrikler | Confusion Matrix ve F1-Score | - |
| **Hata analizi** | Hatalı Tahminler | Isı Haritası | **Grad-CAM** analizi | - |

## Tablo J: Önerilen Model ve Eğitim Konfigürasyonu

| Başlık | Öneri | Alternatifler | Seçim gerekçesi |
| :--- | :--- | :--- | :--- |
| **Model omurgası** | **EfficientNetV2-S** | ResNet50, MobileNet | V1'e göre daha hızlı eğitim ve parametre verimliliği (Google 2021). |
| **Head** | **Coordinate Attention** | Standart Dense | Modelin arka plana değil, yapraktaki hastalık bölgesine odaklanmasını sağlamak. |
| **Kayıp** | **Focal Loss** | Cross Entropy | Dengesiz veri setinde zor örneklere (nadir hastalıklara) odaklanmak. |
| **Optimizer** | **AdamW** | SGD, Adam | Daha iyi weight decay ve genelleme performansı. |
| **LR schedule** | Cosine Annealing | Step Decay | Daha stabil yakınsama ve yerel minimumlardan kaçış. |
| **Batch size** | 32 | 16, 64 | Donanım optimizasyonu. |
| **Epoch** | 30 | 50, 100 | Early stopping ile takip edilecek. |
| **Regularization** | Label Smoothing | Dropout | Aşırı öğrenmeyi (overfitting) engellemek. |
| **Augmentation** | **CutMix & MixUp** | Rotate/Flip | Modern veri artırma teknikleriyle modelin ezberlemesini önlemek. |
| **Seed stratejisi** | 42 | Random | Tekrarlanabilir sonuçlar için sabit seed. |
