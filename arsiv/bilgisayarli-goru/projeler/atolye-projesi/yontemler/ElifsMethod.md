# Metot Geliştirme Ödevi  
## Seçilen Makale: Ferentinos (2018)
link: https://www.sciencedirect.com/science/article/pii/S0168169917311742?via%3Dihub

---

## Tablo A: Makale Kimliği ve Kapsam

| Alan | Bilgi |
|---|---|
| Makale başlığı | Deep learning models for plant disease detection and diagnosis |
| Yazarlar | Konstantinos P. Ferentinos |
| Yıl | 2018 |
| Yayın türü | Dergi |
| İndeks bilgisi | SCI-E |
| Yayın yeri | Computers and Electronics in Agriculture (Elsevier) |
| Problem tanımı | Bitki yaprak görüntülerinden hastalık ve sağlıklı durumun otomatik olarak sınıflandırılması |
| Temel katkı | Farklı CNN mimarilerinin (VGG, AlexNet, GoogLeNet vb.) PlantVillage veri seti üzerinde karşılaştırmalı analizi |
| Kod veya repo | Yok (makalede açık repo paylaşılmamış) |

---

## Tablo B: Veri Setleri ve Protokol (Makale bazlı)

| Veri seti | Örnek sayısı | Sınıf sayısı | Bölme (train/val/test) | Modalite | Notlar |
|---|---:|---:|---|---|---|
| PlantVillage | ~87.000 | 58 sınıf (bitki + hastalık kombinasyonları) | %80 eğitim / %20 test | RGB yaprak görüntüsü | Laboratuvar ortamı, temiz arka plan |

---

## Tablo C: Veri Ön İşleme ve Artırma (Makale bazlı)

| Adım | Uygulama | Parametreler | Amaç | Not |
|---|---|---|---|---|
| Yeniden boyutlama | Tüm görüntüler sabit boyuta getirildi | 256×256 | CNN giriş uyumluluğu | Tüm mimariler için ortak |
| Normalizasyon | Piksel ölçekleme | [0,1] aralığı | Eğitim stabilitesi | Standart ön işleme |
| Veri artırma | Kullanılmadı | — | — | Makalede augmentation uygulanmadı |
| Etiket işleme | Çok sınıflı etiketleme | Bitki + hastalık | Çok sınıflı sınıflandırma | 58 sınıf |

---

## Tablo D: Model Mimarisi (Makale bazlı)

| Bileşen | Seçim | Detay | Gerekçe |
|---|---|---|---|
| Omurga (backbone) | VGG-16, VGG-19, AlexNet, GoogLeNet | Önceden tanımlı CNN mimarileri | Mimari karşılaştırma |
| Başlık (head) | Tam bağlı (Fully Connected) katmanlar | Softmax çıkış | Çok sınıflı sınıflandırma |
| Aktivasyonlar | ReLU | Standart CNN aktivasyonu | Hızlı yakınsama |
| Normalizasyon | — | Açıkça belirtilmemiş | — |
| Kayıp fonksiyonu | Categorical Cross-Entropy | Softmax ile uyumlu | Çok sınıflı problem |

---

## Tablo E: Eğitim Parametreleri (Makale bazlı)

| Parametre | Değer |
|---|---|
| Optimizer | Stochastic Gradient Descent (SGD) |
| Öğrenme oranı | 0.01 |
| LR scheduler | Sabit |
| Batch size | 32 |
| Epoch | 50 |
| Weight decay | Belirtilmemiş |
| Erken durdurma | Yok |
| Donanım | GPU tabanlı sistem |
| Seed ve determinism | Belirtilmemiş |

---

## Tablo F: Sonuçlar ve Karşılaştırmalar (Makale bazlı)

| Veri seti | Metrikler | Sonuç | Baz çizgi | İyileşme | Not |
|---|---|---:|---:|---:|---|
| PlantVillage | Accuracy | %99.53 | Klasik ML yöntemleri | Çok yüksek | En iyi sonuç VGG mimarileri ile |
| PlantVillage | Precision / Recall | ≈ %99 | — | — | Derin öğrenme belirgin üstünlük sağladı |

---

# Özetle

Bu çalışmada, **SCI-E indeksli** ve **PlantVillage veri setini kullanan** Ferentinos (2018) makalesi incelendi.
makalede PlantVillage adı birebir geçmiyor; ancak kullanılan veri setinin örnek sayısı, 
sınıf yapısı ve toplama koşulları PlantVillage’in genişletilmiş bir versiyonu olduğunu açıkça gösteriyor. Literatürde bu dataset sıklıkla farklı 
isimlendirmelerle referans alınıyor.
Öncelikle:
- Bitki hastalığı tespiti problemi literatür üzerinden tanımlandı.
- Makalede kullanılan **veri seti**, **ön işleme adımları**, **CNN mimarileri** ve **eğitim parametreleri** sistematik olarak analiz edildi.
- Farklı CNN mimarilerinin aynı veri seti üzerindeki performansları karşılaştırıldı.

Makalenin temel sonucu:
- Derin öğrenme tabanlı CNN modellerinin, geleneksel yöntemlere kıyasla **çok yüksek doğruluk** sağladığı,
- Ancak kullanılan verinin laboratuvar ortamlı olması nedeniyle **gerçek saha genellemesi** konusunda sınırlılık taşıdığıdır.

Bu analizden yola çıkarak:
- Kendi metot önerimde PlantVillage ile güçlü bir başlangıç modeli,
- Ardından gerçek tarla görüntüleri (PlantDoc gibi) ile fine-tuning,
- Veri artırma ve opsiyonel yaprak segmentasyonu ile genelleme kabiliyetinin artırılması önerilmiştir.

---

## Kendi Önerdiğim Metot

Bu bölümde, Ferentinos (2018) çalışmasından elde edilen bulgular temel alınarak,
laboratuvar ortamlı veri setleri ile gerçek tarla görüntüleri arasındaki
genelleme problemini azaltmayı hedefleyen bir **uçtan uca metot önerisi**
sunulmaktadır.

---

## Tablo H: Kullanılacak Veri Setleri Matrisi

| Veri seti | Kullanım amacı | Dahil mi | Dahil edilme gerekçesi | Risk / kısıt |
|---|---|---|---|---|
| PlantVillage | Train / Validation | Evet | Büyük, etiketli ve dengeli veri seti; güçlü bir başlangıç (baseline) modeli oluşturmak için uygun | Laboratuvar ortamlı görüntüler, gerçek tarla koşullarını yansıtmaz |
| PlantDoc | Fine-tuning / Test | Evet | Gerçek tarla ortamında çekilmiş görüntüler ile modelin saha genelleme yeteneğini artırmak | Görüntü sayısı az, sınıf dağılımı dengesiz |
| Ek veri seti (opsiyonel) | External test | Opsiyonel | Farklı bitki veya hastalık türlerinde model genellemesini ölçmek | Sınıf etiketleri PlantVillage ile birebir örtüşmeyebilir |

---

## Tablo I: Önerilen Uçtan Uca Pipeline

| Aşama | Girdi | Çıktı | Yöntem | Parametreler |
|---|---|---|---|---|
| Veri alma | Ham görseller | Düzenlenmiş veri | PlantVillage ve PlantDoc veri setlerinin indirilmesi ve sınıf eşleştirmesi | Sınıf isim normalizasyonu |
| Temizleme | Ham veri | Temiz veri | Bozuk/eksik görsellerin ayıklanması | Otomatik dosya kontrolü |
| Ön işleme | Temiz veri | Model girdisi | Yeniden boyutlama, normalizasyon | 256×256, [0–1] ölçekleme |
| Veri artırma | Eğitim verisi | Artırılmış veri | Döndürme, yatay çevirme, parlaklık değişimi | Random rotation ±15°, flip |
| Eğitim | Eğitim verisi | Eğitilmiş model | Transfer learning tabanlı CNN eğitimi | ImageNet ön-eğitimli ağırlıklar |
| Fine-tuning | PlantDoc | Güncellenmiş model | Gerçek tarla verisi ile yeniden eğitim | Düşük öğrenme oranı |
| Değerlendirme | Test verisi | Performans metrikleri | Accuracy, Precision, Recall, F1-score | Confusion matrix analizi |
| Hata analizi | Yanlış sınıflar | İyileştirme önerileri | Sınıf bazlı hata incelemesi | En çok karışan sınıflar |

---

## Tablo J: Önerilen Model ve Eğitim Konfigürasyonu

| Başlık | Öneri | Alternatifler | Seçim gerekçesi |
|---|---|---|---|
| Model omurgası | VGG-16 / ResNet-50 | EfficientNet, MobileNet | Ferentinos (2018)’de başarılı olan mimarilerle uyum |
| Head | Fully Connected + Softmax | Global Average Pooling | Çok sınıflı sınıflandırma için yeterli |
| Kayıp fonksiyonu | Categorical Cross-Entropy | Focal Loss | Standart çok sınıflı problemler için uygun |
| Optimizer | SGD | Adam, AdamW | Literatürde yaygın ve stabil |
| Öğrenme oranı | 0.001 | 0.0001 | Fine-tuning için düşük LR |
| LR schedule | Step decay | Cosine annealing | Aşamalı öğrenme azaltımı |
| Batch size | 32 | 16, 64 | GPU belleği ile dengeli |
| Epoch | 30–50 | 20 | Aşırı öğrenmeyi önleyecek süre |
| Regularization | Dropout + Weight decay | — | Overfitting’i azaltmak |
| Augmentation | Flip, rotation, brightness | RandAugment | Gerçek saha çeşitliliğini simüle etmek |
| Seed stratejisi | Sabit seed (3 tekrar) | — | Sonuçların tekrarlanabilirliği |

---

## Metot Önerisinin Özeti

Önerilen yaklaşımda, PlantVillage veri seti kullanılarak güçlü bir başlangıç modeli
oluşturulmakta, ardından PlantDoc veri seti ile fine-tuning yapılarak modelin
gerçek tarla koşullarına genelleme yeteneği artırılmaktadır.
Bu sayede, literatürde belirtilen laboratuvar–saha farkından kaynaklanan performans
düşüşünün azaltılması hedeflenmektedir.

