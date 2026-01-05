# 🌿 PlantVillage Geniş Kapsamlı Bitki Hastalıkları Veri Seti

## 🔍 Veri Seti Özeti

Bu veri seti, dünya çapında bilim insanları ve geliştiriciler tarafından bitki hastalıklarının otomatik teşhisi için kullanılan, kontrollü koşullar altında toplanmış geniş bir görüntü koleksiyonudur. Bilgisayarlı Görü (CV) ile sınıflandırma projeleri için idealdir.

## 1. 🔗 Veri Seti Kaynak Bilgileri

* **Platform:** Kaggle
* **Tam Kaynak Linki:** [https://www.kaggle.com/datasets/mohitsingh1804/plantvillage](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage)
* **Orijinal Kaynak:** PlantVillage Projesi (Penn State Üniversitesi)
* **Lisans Tipi:** Genellikle Creative Commons (CC) lisansları altındadır. (Kullanım detayları için Kaggle sayfasını kontrol ediniz.)

## 2. 📊 İçerik Detayları

| Kategori | Değer/Açıklama |
| :--- | :--- |
| **Toplam Görüntü Sayısı** | Yaklaşık **54.300** adet |
| **Kapsanan Bitki Türü** | 14 farklı ürün (Örn: Elma, Domates, Biber, Patates, Çilek vb.) |
| **Sınıflandırma Etiketi Sayısı** | **38** sınıf (Farklı bitkilerin hastalıklı ve sağlıklı durumlarını kapsar) |
| **Görüntü Formatı** | JPEG |
| **Toplama Koşulu** | Kontrollü laboratuvar koşulları (Arka plan genellikle tek renktir) |

## 3. 🎯 Proje Amacı ve Potansiyel Kullanım Alanları

Bu veri seti, projemizdeki Bilgisayarlı Görü görevine **mükemmel** uyum sağlar.

* **Sınıflandırma:** 38 farklı hastalık ve sağlıklı yaprak durumunu yüksek doğrulukla sınıflandırmak.
* **Transfer Öğrenme:** Veri setinin büyüklüğü ve temiz yapısı sayesinde, hazır Convolutional Neural Network (CNN) modellerini (ResNet, VGG) eğitmeye çok uygundur.
* **Temel Proje:** Bu veri seti, projemiz için hızlıca bir temel model (Baseline Model) oluşturmak için kullanılabilir.

## 4. ✍️ Notlar ve Ek Bilgiler

* Görüntülerin kontrollü koşullarda çekilmiş olması, modelin doğal ortamdaki görüntülere genelleme yeteneğini (generalization) düşürebilir. Proje ilerledikçe bu durum tartışılmalıdır.
* Veri setinin temizlenmiş ve etiketlenmiş yapısı, ön işleme (preprocessing) yükünü önemli ölçüde azaltmaktadır.
