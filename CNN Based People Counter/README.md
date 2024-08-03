## CNN Based People Counter

Görev bir görüntüde kaç kişi olduğunu hesaplayan bir CNN modelini geliştirmeyi kapsamaktadır. Etkinlik katılılımcı sayısı, sınıf mevcudu gibi esnek görevlerde kullanılabilecek bir model geliştirmelisiniz. Girdi görüntüleri gerçek zamanlı bir video dan olabileceği gibi standart bir resim dosyasını ekleyerekte olabilmeli. Görev süresi 3 haftadır. Bitiş tarihi: 23 Ağustos Cuma günü. Başarılar dilerim.

## Veri Seti
Önerdiğimiz veri seti COCO. Veri setini kendi web sitesinden indirebilirsiniz. Etiketlerinide json dosyası olarak mevcut. Json'da etiket adı ve görseldeki etiket koordinatları var. Sizler json için veri özelleştirme kodu yazmalısınız. 90 bine yakın görsel içerisinden kategorisi "person" olan dosyaları kullanmalısınız ve CNN ile object detection modeli kurmalısınız.

## Modelleme

Kendi CNN Object Detection modelinizi tasarlayabilir veya YOLO gibi hazır modelleri kullanabilirsiniz. Görüntüde kişi sayısını saymak için OPENCV ile detection sayısını(resimde kaç kişi olduğunu) ekranın küçük bir köşesine yazdırmalısınız. Bu gerçek zamanlı bir girdi için anlık olmalı ve FPS değerini de ekranın diğer köşesine yazdırmalısınız.  

## Değerlendirme

Modellerinizi kendi test veriniz ile test edip Accurcy, F1 Skoru ve mAP ile değerlendirmelisiniz.
