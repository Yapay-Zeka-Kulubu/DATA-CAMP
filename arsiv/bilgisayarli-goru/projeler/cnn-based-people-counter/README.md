## CNN Based People Counter

Görev bir görüntüde kaç kişi olduğunu hesaplayan bir CNN modelini geliştirmeyi kapsamaktadır. Etkinlik katılılımcı sayısı, sınıf mevcudu gibi esnek görevlerde kullanılabilecek bir model geliştirmelisiniz. Girdi görüntüleri gerçek zamanlı bir video dan olabileceği gibi standart bir resim dosyasını ekleyerekte olabilmeli. Görev süresi 3 haftadır. Bitiş tarihi: 30 Ağustos Cuma günü. Başarılar dilerim.

## Veri Seti
Önerdiğimiz veri seti COCO. Veri setini kendi web sitesinden indirebilirsiniz. Etiketlerinide json dosyası olarak mevcut. Json'da etiket adı ve görseldeki etiket koordinatları var. Sizler json için veri özelleştirme kodu yazmalısınız. 90 bine yakın görsel içerisinden kategorisi "person" olan dosyaları kullanmalısınız ve CNN ile object detection modeli kurmalısınız.

## Modelleme

Kendi CNN Object Detection modelinizi tasarlayabilir veya YOLO gibi hazır modelleri kullanabilirsiniz. Görüntüde kişi sayısını saymak için OPENCV ile detection sayısını(resimde kaç kişi olduğunu) ekranın küçük bir köşesine yazdırmalısınız. Bu gerçek zamanlı bir girdi için anlık olmalı ve FPS değerini de ekranın diğer köşesine yazdırmalısınız.  

## Değerlendirme

Modellerinizi kendi test veriniz ile test edip Accurcy, F1 Skoru ve mAP ile değerlendirmelisiniz.

# Proje Geliştirme Detayları

## Json ile veri seti ayrıştırma
COCO verisi görsellerin bilgilerini json dosyasında tutar. train2014 veri seti için instances_train2014.json dosyasında görsellerin dosya adlarına karşılık gelen kategori isimleri ve etiket kordinatları bulunmaktadır. Kategoriler, koordinatların gösterdiği dörtgen alanın içerisinde bulunan görselin sınıf ismidir. Sizler json dosyasını kategori ismi person olan görselleri ayrıştırıp eğiteceğiniz modele göre etiketlerini xml,txt veya başka bir yapıda veri seti klasörünüze kayıt etmelisiniz.

## Model Eğitimi
YOLO, Faster-R-CNN veya kendi tasarladığınız object detection modeline uygun veri seti yapısını oluşturmalısınız. Görsellerin etiket koordinatları kendi orjinal boyutlarına özgü eğer her eğitimde statik boyut kabul eden bir modeliniz var ise görsellerii bir boyuta eşitlemeli, etiket koordinatlarınızı ise boyut değişiliğine göre normalize etmelisiniz. Aksi durumda etiketlerinizde sapma oluşacaktır. Eğer modeliiz dinamik bouylara karşı duyarlı ise direkt eğitim yapabilirsiniz.

## Küçük Bir Tavsiye
CNN mimarisini ve nesne tespit görevini teorik olarak anlamadan görevi gerçekleştirmeniz sizler için bir şey ifade etmeyecektir.

