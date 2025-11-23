## 🚗 Hafta Ödevi – Araç Kamerası ile Şerit Belirleme

Bu haftaki ödeviniz:

Bir araç ön kamerasından alınmış videoyu OpenCV ile açarak video görüntüsünü anlık olarak önce **grayscale**, daha sonra **siyah-beyaz dönüşümü** yapmak.  
OpenCV’de siyah-beyaz dönüşümünün adı: **Binary Thresholding (cv2.threshold)**  
Bu görüntü üzerinde **Canny** kenar algılama uygulayarak şeritleri belirginleştirmek.

### Yapılacaklar
- Videoyu OpenCV ile açmak  
- Her karede şu adımları uygulamak:  
  - Grayscale dönüşümü  
    `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)`
  - Siyah-beyaz dönüşümü  
    `cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)`
  - Canny kenar algılama  
    `cv2.Canny(binary, 50, 150)`
  - Şeritleri belirgin hale getirmek

### Opsiyonel
- İlgi alanı (ROI) kırparak sadece yola odaklanmak  
- İlgi alanı alınmış yolun perspektif dönüşümünü (Bird’s-eye view) almak  
  `cv2.warpPerspective()`
