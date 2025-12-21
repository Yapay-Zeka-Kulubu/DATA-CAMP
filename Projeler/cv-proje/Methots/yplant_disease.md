### Tablo A: Makale Kimliği ve Kapsam
| Alan | Bilgi |
|---|---|
| Makale başlığı | Using Deep Learning for Image-Based Plant Disease Detection  |
| Yazarlar | Sharada P. Mohanty, David P. Hughes, Marcel Salathé|
| Yıl | 2016 |
| Yayın türü | Dergi (Frontiers in Plant Science) |
| İndeks bilgisi | SCI-E |
| Problem tanımı | Akıllı telefonlar aracılığıyla bitki hastalıklarının yaprak fotoğraflarından otomatik teşhis edilmesi. |
| Temel katkı | PlantVillage veri setini kullanarak derin öğrenme mimarilerinin (CNN) bitki hastalıkları üzerindeki başarısını kanıtlamak. |
| Kod veya repo | [https://github.com/spMohanty/PlantVillage-Datase](https://github.com/salathegroup/plantvillage_deeplearning_paper_dataset)        |

### Tablo B: Veri Setleri ve Protokol (Makale bazlı)
| Veri seti | Örnek sayısı | Sınıf sayısı | Bölme (train/val/test) | Modalite | Notlar |
|---|---:|---:|---|---|---|
| PlantVillage |54,306  |38 |%80 Train - %20 Test | RGB / Grayscale  | 14 farklı ürün türü içerir. |

### Tablo C: Veri Ön İşleme ve Artırma (Makale bazlı)
| Adım | Uygulama | Parametreler | Amaç | Not |
|---|---|---|---|---|
| Normalizasyon |Mean Subtraction  |Mean: [104, 117, 123] (RGB kanalları için)  |Gradyanların daha hızlı yakınsamasını sağlamak.  |Veri seti ortalaması her pikselden çıkarılmıştır.  |
| Yeniden boyutlama |Bilinear Interpolation |256 x 256 piksel |Görüntüleri standart boyuta getirmek.  |Modeller (AlexNet/GoogLeNet) sabit boyutlu girdi bekler.|
| Augmentation |Bu çalışmada temel deneylerde veri artırma sınırlı tutulmuş, modelin ham veri üzerindeki ayırt ediciliği ölçülmüştür.  |N/A  |Ham verinin ham başarısını ölçmek.  |Makale, artırma yapmadan %99 başarıya odaklanmıştır.  |
| Etiket işleme |Integer/One-Hot Encoding  |38 Farklı Sınıf (Kategori)  |Kategorik veriyi sayısal vektöre dönüştürmek.  |Her "Bitki+Hastalık" kombinasyonu tek bir sayısal ID almıştır.  |

### Tablo D: Model Mimarisi (Makale bazlı)
| Bileşen | Seçim | Detay | Gerekçe |
|---|---|---|---|
| Omurga (backbone) |GoogLeNet (Inception v1)  |22 katman derinliğinde, Inception modülleri içeren yapı.  |Daha derin bir ağ olmasına rağmen AlexNet'ten daha az parametre ile daha yüksek doğruluk sağlaması.  |
| Başlık (head) |Softmax Layer  |Global average pooling sonrası Softmax tabanlı sınıflandırma katmanı (38 sınıf)  |Her bir görselin 38 farklı hastalık sınıfından hangisine ait olduğunu olasılıksal olarak belirlemek.  |
| Aktivasyonlar |ReLU  |f(x) = \max(0, x) fonksiyonu.  |Doğrusal olmayan özellikleri öğrenmek ve gradyan kaybolması (vanishing gradient) problemini azaltmak.  |
| Normalizasyon |LRN (sınırlı kullanım) | Erken katmanlarda yerel cevap normalizasyonu. | Aktivasyon ölçeklerini dengelemek ve eğitimi stabilize etmek.  |
| Kayıp fonksiyonu |Multinomial Logistic Loss  |Cross-Entropy varyasyonu olan çok sınıflı lojistik kayıp.  |Çok sınıflı sınıflandırma problemlerinde tahmin edilen olasılık ile gerçek etiket arasındaki farkı minimize etmek. |

### Tablo E: Eğitim Parametreleri (Makale bazlı)
| Parametre | Değer |
|---|---|
| Optimizer |SGD (Stochastic Gradient Descent)  |
| Öğrenme oranı |0.01(başlangıç değeri)  |
| LR scheduler |Step Decay  |
| Batch size |AlexNet: 100, GoogLeNet: 32  |
| Epoch |30 - 50  |
| Weight decay |0.0005 | 
| Erken durdurma |Uygulanmadı|.
| Donanım |NVIDIA Titan X GPU  |
| Seed ve determinism |Belirtilmemiş  |

### Tablo F: Sonuçlar ve Karşılaştırmalar (Makale bazlı)
| Veri seti | Metrikler | Sonuç | Baz çizgi | İyileşme | Not |
|---|---|---:|---:|---:|---|
|PlantVillage | Accuracy (Doğruluk), |%99.35  |%98.20  |%1.15  |Baz çizgi olarak AlexNet tabanlı CNN performansı alınmıştır.GoogLeNet mimarisi, AlexNet’e kıyasla yaklaşık %1.15 doğruluk artışı sağlamıştır.  |
|PlantVillage | F1 Score|0.9934   |0.9815  |0.0119  |Makale ağırlıklı olarak accuracy raporlamaktadır; F1 skoru ikincil değerlendirme olarak sunulmuştur.  |
                          

## Kendi Önerdiğiniz Metot İçin Zorunlu Tablolar
Bu tablolar, “benim metot önerim” kısmında doldurulacaktır.

### Tablo H: Kullanılacak Veri Setleri Matrisi
| Veri seti | Kullanım amacı | Dahil mi | Dahil edilme gerekçesi | Risk / kısıt |
|---|---|---|---|---|
| PlantVillage | Ana Model Eğitimi | Evet |38 farklı sınıf ve 54.300+ görsel ile literatürdeki en kapsamlı ve standart veri setidir.  |Laboratuvar ortamında çekildiği için gerçek tarla koşullarında başarı düşebilir.  |
| Indoor Plant Disease Detection|Modelin Genelleştirilmesi  | Evet |İç mekan bitkileri ve farklı arka planlar içerdiği için modelin farklı ortamlardaki başarısını test eder.  |Sınıf sayısının ana veri setinden az olması nedeniyle sadece ortak sınıflarda kullanılabilir.  |

### Tablo I: Önerilen Uçtan Uca Pipeline
| Aşama | Girdi | Çıktı | Yöntem | Parametreler |
|---|---|---|---|---|
| Veri alma |GitHub / Cloud  |Ham .jpg Görüntüler  |HTTP / Git tabanlı veri çekme |54.306 adet görsel  |
| Temizleme |Ham Görüntüler  |Filtrelenmiş Veri  |OpenCV ile format ve bozuk dosya kontrolü  |RGB mod kontrolü, bozuk dosya eleme  |
| Ön işleme |Temiz Resimler  |Girdi Tensörleri  |Torchvision Transforms  |Resize: 224x224, Normalize: ImageNet mean/std  |
| Eğitim |Eğitim Seti (%80)  |Eğitilmiş Model (.pth)  |Transfer Learning (Pre-trained ResNet)  |50 Epoch, Early Stopping (patience=7)  |
| Değerlendirme |Test Seti (%20) |Metrik Raporu  |Scikit-Learn Classification Report  |Accuracy, F1-Score, Confusion Matrix  |
| Hata analizi |Yanlış Tahminler  |Isı Haritaları  |Grad-CAM Görselleştirme  |Sınıf bazlı ısı haritası (Activation maps)  |

### Tablo J: Önerilen Model ve Eğitim Konfigürasyonu
| Başlık | Öneri | Alternatifler | Seçim gerekçesi |
|---|---|---|---|
| Model omurgası |ResNet-50  |VGG16, MobileNet  |"Skip Connections" sayesinde daha derin katmanlarda bile "vanishing gradient" sorunu yaşatmaz ve daha yüksek doğruluk sunar.  |
| Head |Fully Connected + Softmax  |Global Average Pooling  |Özellik haritalarını uzamsal olarak özetleyerek parametre sayısını azaltır ve aşırı öğrenmeyi önler.  |
| Kayıp |Cross-Entropy Loss  |Focal Loss  |Çok sınıflı sınıflandırmada olasılık dağılımını ölçmek için en kararlı yöntemdir.  |
| Optimizer |Adam  |SGD, RMSprop  |Öğrenme oranını her parametre için otomatik ayarlar (adaptive), SGD'den daha hızlı yakınsar.  |
| LR schedule |Cosine Annealing  |Step Decay  |Öğrenme oranını kademeli ve pürüzsüz şekilde azaltarak eğitimin son aşamalarında daha kararlı yakınsama sağlar.  |
| Batch size |64  |16, 32, 128  |GPU bellek kullanımı ve gradyan gürültüsü arasındaki en ideal dengeyi sağlar.  |
| Epoch |50  |30, 100  |Modelin doyuma ulaşması için yeterli bir süredir (Early Stopping ile desteklenir). |
| Regularization |Dropout (0.3)  |L1/L2 Norm  |Eğitim sırasında rastgele nöronları kapatarak modelin ezberlemesini (overfitting) engeller.  |
| Augmentation |AutoAugment  |Sadece Flip  |Parlaklık, döndürme ve kesme işlemlerini otomatik yaparak modelin tarladaki farklı ışık koşullarına uyumunu sağlar. |
| Seed stratejisi |42 (Fixed)  |Random  |Deneylerin her seferinde aynı sonuçları vermesi (tekrarlanabilirlik) için sabitlenmiştir. |


