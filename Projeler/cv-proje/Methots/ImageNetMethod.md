Deep Residual Learning for Image Recognition
Yazar: Kaiming He et al.
Veri seti: ImageNet (ILSVRC 2012)


## Tablo A: Makale Kimliği ve Kapsam
| Alan           | Bilgi                                                                                                                   |
| -------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Makale başlığı | Deep Residual Learning for Image Recognition                                                                            |
| Yazarlar       | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun                                                                       |
| Yıl            | 2016                                                                                                                    |
| Yayın türü     | Konferans (CVPR)                                                                                                        |
| İndeks bilgisi | SCI-E                                                                                                                   |
| Problem tanımı | Çok derin sinir ağlarının eğitimi sırasında ortaya çıkan **degradation problem** (derinlik arttıkça doğruluğun düşmesi) |
| Temel katkı    | Residual (skip) bağlantılar ile çok derin ağların kararlı ve etkili şekilde eğitilebilmesi                              |
| Kod veya repo  | Var (resmi ve topluluk implementasyonları mevcut)                                                                       |



## Tablo B: Veri Setleri ve Protokol
| Veri seti              | Örnek sayısı | Sınıf sayısı | Bölme (train/val/test) | Modalite    | Notlar                                            |
| ---------------------- | -----------: | -----------: | ---------------------- | ----------- | ------------------------------------------------- |
| ImageNet (ILSVRC 2012) |       ~1.28M |         1000 | ~1.28M / 50k / 100k    | RGB görüntü | Büyük ölçekli benchmark, gerçek dünya görüntüleri |



## Tablo C: Veri Ön İşleme ve Artırma
| Adım              | Uygulama               | Parametreler      | Amaç                     | Not                       |
| ----------------- | ---------------------- | ----------------- | ------------------------ | ------------------------- |
| Normalizasyon     | Mean–Std normalization | ImageNet mean/std | Dağılımı stabilize etmek | CNN eğitimi için standart |
| Yeniden boyutlama | Resize + Crop          | 256 → 224         | Sabit girdi boyutu       | Hesaplama verimliliği     |
| Augmentation      | Random crop, flip      | Horizontal flip   | Overfitting azaltma      | Eğitim setine özel        |
| Etiket işleme     | One-hot / index        | 1000 sınıf        | Çok sınıflı öğrenme      | Softmax ile uyumlu        |



## Tablo D: Model Mimarisi (Makale bazlı)
| Bileşen           | Seçim                               | Detay               | Gerekçe                         |
| ----------------- | ----------------------------------- | ------------------- | ------------------------------- |
| Omurga (backbone) | ResNet-50 / ResNet-101 / ResNet-152 | Çok katmanlı CNN    | Derin ağlarda performans artışı |
| Başlık (head)     | Global Avg Pool + FC                | 1000 çıkış          | Parametre sayısını azaltır      |
| Aktivasyonlar     | ReLU                                | Katman sonrası      | Non-lineer temsil               |
| Normalizasyon     | Batch Normalization                 | Her konv katmanında | Eğitim stabilitesi              |
| Kayıp fonksiyonu  | Cross-Entropy Loss                  | Softmax tabanlı     | Çok sınıflı problem             |



## Tablo E: Eğitim Parametreleri
| Parametre           | Değer                           |
| ------------------- | ------------------------------- |
| Optimizer           | SGD                             |
| Öğrenme oranı       | 0.1 (step decay)                |
| LR scheduler        | Step decay                      |
| Batch size          | 256                             |
| Epoch               | ~90                             |
| Weight decay        | 1e-4                            |
| Erken durdurma      | Kullanılmadı                    |
| Donanım             | Çoklu GPU (NVIDIA)              |
| Seed ve determinism | Kısmi (tam deterministik değil) |



## Tablo F: Sonuçlar ve Karşılaştırmalar
| Veri seti | Metrikler      |  Sonuç |       Baz çizgi | İyileşme | Not                      |
| --------- | -------------- | -----: | --------------: | -------: | ------------------------ |
| ImageNet  | Top-1 Accuracy | ~75.3% | VGG-19 (~72.7%) |    +2.6% | Derinlik arttıkça kazanç |
| ImageNet  | Top-5 Accuracy | ~92.2% |       GoogLeNet |        + | Benchmark SOTA           |



## Tablo H: Kullanılacak Veri Setleri Matrisi
| Veri seti              | Kullanım amacı       | Dahil mi  | Dahil edilme gerekçesi | Risk / kısıt                |
| ---------------------- | -------------------- | --------- | ---------------------- | --------------------------- |
| ImageNet (ILSVRC 2012) | Train / Val / Test   | Evet      | Benchmark ve genelleme | Donanım ihtiyacı yüksek     |
| Tiny ImageNet          | Ön deney / ablation  | Evet      | Hızlı prototipleme     | Daha düşük varyasyon        |
| CIFAR-100              | Mimari karşılaştırma | Opsiyonel | Küçük ölçek test       | ImageNet’ten farklı dağılım |



## Tablo I: Önerilen Uçtan Uca Pipeline
| Aşama         | Girdi          | Çıktı             | Yöntem              | Parametreler   |
| ------------- | -------------- | ----------------- | ------------------- | -------------- |
| Veri alma     | Ham görüntüler | Temiz veri        | ImageFolder         | —              |
| Temizleme     | Görüntüler     | Filtrelenmiş veri | Bozuk veri ayıklama | Boyut kontrolü |
| Ön işleme     | Görüntüler     | Tensor            | Resize + Normalize  | 224×224        |
| Eğitim        | Tensor         | Eğitilmiş model   | ResNet + SGD        | lr, batch      |
| Değerlendirme | Test seti      | Metrikler         | Accuracy, F1-macro  | sklearn        |
| Hata analizi  | Tahminler      | Hata profili      | Confusion matrix    | Sınıf bazlı    |



## Tablo J: Önerilen Model ve Eğitim Konfigürasyonu
| Başlık          | Öneri                 | Alternatifler     | Seçim gerekçesi       |
| --------------- | --------------------- | ----------------- | --------------------- |
| Model omurgası  | ResNet-18 / ResNet-50 | EfficientNet, ViT | Kanıtlanmış stabilite |
| Head            | GAP + FC              | Attention head    | Basit ve etkili       |
| Kayıp           | Cross-Entropy         | Focal Loss        | Dengeli sınıflar      |
| Optimizer       | Adam / SGD            | RMSprop           | Hız ve stabilite      |
| LR schedule     | Step / Cosine         | ReduceLROnPlateau | Öğrenme kontrolü      |
| Batch size      | 32–256                | —                 | Donanım uyumu         |
| Epoch           | 20–90                 | —                 | Overfitting kontrolü  |
| Regularization  | Weight decay          | Dropout           | Genelleme             |
| Augmentation    | Flip + Rotation       | AutoAugment       | Basitlik              |
| Seed stratejisi | Fixed seed            | Multi-seed        | Tekrarlanabilirlik    |


##
