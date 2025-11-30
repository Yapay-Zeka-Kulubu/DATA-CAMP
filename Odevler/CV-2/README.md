# Ödev – Bilgisayarlı Görü (Hafta 2)

Bu ödevde, bir araç ön kamerasından alınmış yol videosu üzerinde çalışarak **şeritlerin orta noktasını**, **aracın orta noktasını** ve bunlar arasındaki **şeritten sapma miktarını** hesaplayan bir OpenCV programı yazmanız beklenmektedir.

## Görev Tanımı

Bir yol videosunu kullanarak:

1. Videoyu kare kare (frame) okuyun.
2. Her kare için:
   - Yolu ve şerit çizgilerini belirginleştirecek uygun bir **ön işleme (pre-processing)** adımı uygulayın  
     (örneğin: `grayscale`, `blur`, `Canny` kenar bulma vb.).
   - Sadece yolun bulunduğu bölgeyi incelemek için bir **ilgi alanı (Region of Interest – ROI)** tanımlayın.
   - Şerit çizgilerini tespit edin  
     (örneğin: `HoughLinesP` veya kendi şerit tespit mantığınız).
3. Tespit edilen şerit çizgilerine göre:
   - **Şerit merkezini** (şeritlerin orta noktası) hesaplayın.
   - **Aracın görüntü içindeki merkezini** (frame orta noktası varsayabilirsiniz) belirleyin.
   - Araç merkezi ile şerit merkezi arasındaki **piksel cinsinden sapma miktarını** hesaplayın.
4. Her kare üzerinde:
   - Şerit çizgilerini,
   - Şerit merkezini,
   - Araç merkezini,
   - Sapma miktarını (örneğin: “Offset: +15 px sağa” gibi)
   
   görsel olarak çizin ve ekranda gösterin.

## Beklenen Çıktılar

- Çalışan bir **Python + OpenCV** programı:
  - Yol videosunu okuyup gerçek zamanlı (veya kare kare) işleyen
  - Şerit merkezini ve araç merkezini tespit eden
  - Şeritten sapma miktarını hesaplayıp ekranda gösteren
- İsteğe bağlı olarak:
  - Sapma miktarını her kare için konsola da yazdırabilirsiniz.
  - Sapma değerini zamanla gösteren basit bir grafik/log dosyası oluşturabilirsiniz.

