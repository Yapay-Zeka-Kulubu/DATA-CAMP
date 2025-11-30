# Telco Customer Churn Analysis

Bu proje ödevi, Telco Customer Churn verisi kullanılarak müşteri kayıp analizi yapmayı amaçlamaktadır. Analiz, veri temizleme, keşifsel veri analizi, özellik mühendisliği ve tahmin modeli oluşturmayı içermektedir. Aşağıda bu analizin tamamlanması için atılması gereken adımlar bulunmaktadır.

### Not: Tüm sorularınız için Discord soru cevap kısmına yazabilirsiniz. Ödev som teslim tarihi 31 temmuz çarşamba. Proje sonunda herkes projelerinin sunumunu gerçekleştirecektir.

## İçindekiler
1. [Giriş](#giriş)
2. [Veri Seti](#veri-seti)
3. [Veri Temizleme](#veri-temizleme)
4. [Keşifsel Veri Analizi](#keşifsel-veri-analizi)
5. [Özellik Mühendisliği](#özellik-mühendisliği)
6. [Modelleme](#modelleme)
7. [Değerlendirme](#değerlendirme)



## Giriş
Müşteri kaybı (churn), müşterilerin bir işletme ile iş yapmayı bırakma oranını ifade eder. Bu analiz, müşteri kaybını etkileyen faktörleri belirlemeyi ve müşterilerin kayıp olasılığını sınıflandıracak bir tahmin modeli oluşturmayı amaçlamaktadır.

## Veri Seti
Telco Customer Churn veri seti, [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn) üzerinde bulunabilir. Veri seti aşağıdaki bilgileri içerir:

- **customerID**: Müşteri ID'si
   - Her müşteri için benzersiz bir kimlik numarasıdır.
- **gender**: Müşterinin cinsiyeti
   - Müşterinin erkek (Male) veya kadın (Female) olduğunu belirtir.
- **SeniorCitizen**: Müşterinin kıdemli vatandaş olup olmadığı
   - Müşterinin kıdemli vatandaş (Senior Citizen) olup olmadığını gösterir; 1, kıdemli vatandaş olduğunu, 0 ise olmadığını belirtir.
- **Partner**: Müşterinin bir partneri olup olmadığı
   - Müşterinin bir partneri (eş veya hayat arkadaşı) olup olmadığını gösterir; "Yes" partneri olduğunu, "No" partneri olmadığını belirtir.
- **Dependents**: Müşterinin bağımlıları olup olmadığı
   - Müşterinin bağımlıları (çocuklar, yaşlılar vb.) olup olmadığını gösterir; "Yes" bağımlıları olduğunu, "No" bağımlıları olmadığını belirtir.
- **tenure**: Müşterinin şirkette kaldığı ay sayısı
   - Müşterinin şirkette kaldığı toplam ay sayısını belirtir.
- **PhoneService**: Müşterinin telefon hizmeti olup olmadığı
   - Müşterinin telefon hizmeti alıp almadığını gösterir; "Yes" telefon hizmeti olduğunu, "No" telefon hizmeti olmadığını belirtir.
- **MultipleLines**: Müşterinin birden fazla hattı olup olmadığı
   - Müşterinin birden fazla telefon hattı olup olmadığını gösterir; "Yes" birden fazla hattı olduğunu, "No" birden fazla hattı olmadığını, "No phone service" telefon hizmeti olmadığını belirtir.
- **InternetService**: Müşterinin internet hizmet sağlayıcısı
   - Müşterinin kullandığı internet hizmet sağlayıcısını belirtir; "DSL" DSL hizmeti olduğunu, "Fiber optic" fiber optik hizmeti olduğunu, "No" internet hizmeti olmadığını belirtir.
- **OnlineSecurity**: Müşterinin çevrimiçi güvenliği olup olmadığı
   - Müşterinin çevrimiçi güvenlik hizmeti alıp almadığını gösterir; "Yes" çevrimiçi güvenliği olduğunu, "No" çevrimiçi güvenliği olmadığını, "No internet service" internet hizmeti olmadığını belirtir.

## Veri Temizleme

Veri temizleme, veri analizi ve modelleme sürecinin kritik bir aşamasıdır. Bu adım, verideki hataları düzeltmeyi, eksik veya yanlış verileri ele almayı ve veriyi modelleme için uygun hale getirmeyi içerir. Aşağıda, Telco Customer Churn veri setini temizlemek için adım adım yapmanız gereken işlemler açıklanmıştır.

### Veri Yükleme
Veri setini pandas kullanarak yükleyin.

1. **CSV Dosyasını Yükleme**: Pandas kütüphanesi kullanılarak veri seti bir DataFrame'e yüklenir. Bu adım, veriyi analiz edebilmek ve temizleme işlemlerine başlayabilmek için gereklidir.

### Eksik Değerleri Ele Alma
Veri setinde eksik değer olup olmadığını kontrol edin ve bu değerleri uygun şekilde ele alın.

1. **Eksik Değerlerin Kontrolü**: Veri setindeki eksik değerleri belirlemek için çeşitli yöntemler kullanılır. `isnull()` ve `sum()` gibi fonksiyonlar, her sütunda kaç eksik değer olduğunu tespit etmek için kullanılır.
2. **Eksik Değerleri Doldurma veya Kaldırma**: Eksik değerler belirlendikten sonra, bu değerlerin nasıl ele alınacağına karar verilmelidir:
   - **Doldurma (Imputation)**: Eksik değerler, ortalama, medyan veya mod gibi uygun bir değeri kullanarak doldurulabilir. Bu yöntem, veri kaybını önler ve modelin performansını artırabilir.
   - **Kaldırma**: Eksik değerlerin bulunduğu satırları veya sütunları tamamen kaldırabilirsiniz. Bu yöntem, özellikle eksik değerlerin az olduğu durumlarda kullanılabilir.

### Veri Tiplerini Dönüştürme
Her sütun için doğru veri tiplerinin kullanıldığından emin olun.

1. **Veri Tiplerinin Kontrolü ve Dönüştürülmesi**: Her sütunun doğru veri tipine sahip olduğundan emin olmak için veri tipleri kontrol edilir. Sayısal değer olması gereken sütunlar uygun tipe dönüştürülür. Örneğin, sayısal olması gereken ancak metin formatında olan sütunlar sayısal değere dönüştürülmelidir. Bu işlem, analiz ve modelleme aşamalarında doğruluğu artırır.

### Modelin Başarısını Artırma Potansiyeli Olan Yöntemler

#### Eksik Değerleri Doldurma
Eksik değerleri uygun şekilde doldurmak, modelin doğruluğunu artırabilir. Aşağıdaki yöntemleri kullanabilirsiniz:
- **Ortalama ile Doldurma**: Sürekli veriler için, eksik değerler ortalama ile doldurulabilir. Bu yöntem, verideki genel eğilimi korur.
- **Medyan ile Doldurma**: Medyan, aşırı değerlere duyarlı olmadığından eksik değerleri doldurmak için kullanışlıdır.
- **Mod ile Doldurma**: Kategorik veriler için, eksik değerler en sık rastlanan değer (mod) ile doldurulabilir.

#### Veri Tipi Dönüştürme
Veri tiplerini doğru şekilde ayarlamak, veri analizinde ve modellemede büyük fark yaratır:
- **Sayısal Dönüşüm**: Metin formatındaki sayısal verileri sayısal formata dönüştürmek, sayısal analiz ve istatistiksel işlemler için gereklidir.
- **Kategorik Dönüşüm**: Kategorik verileri uygun şekilde etiketlemek (örneğin, 0 ve 1 ile) veya one-hot encoding gibi yöntemlerle sayısal hale getirmek, makine öğrenimi modelleri için veriyi daha uygun hale getirir.

Bu yöntemler, veri temizleme sürecini tamamlayarak veri setini analiz ve modelleme için uygun hale getirecektir. Her adımda dikkatli ve özenli olmak, elde edilen sonuçların doğruluğunu ve güvenilirliğini artıracaktır.

## Keşifsel Veri Analizi

Keşifsel Veri Analizi (Exploratory Data Analysis - EDA), veri setinin yapısını, dağılımını ve ana özelliklerini anlamak için yapılan ilk analizdir. Bu aşama, veri setinin genel özelliklerini belirlemeye ve modelleme sürecine hazırlık yapmaya yardımcı olur. Aşağıda, Telco Customer Churn veri seti için yapılması gereken keşifsel veri analizi adımları detaylandırılmıştır.

### Veri Dağılımını İnceleme
Her sütunun istatistiksel özetini çıkarın ve veri dağılımını inceleyin.

1. **Temel İstatistikler**: Her sütun için ortalama (mean), medyan (median), standart sapma (standard deviation), minimum (min) ve maksimum (max) değerler gibi temel istatistikleri inceleyin. Bu istatistikler, veri setinin genel dağılımını ve merkezi eğilimlerini anlamanıza yardımcı olur.

    - **Ortalama (Mean)**: Verinin merkezi eğilim noktasını belirler.
    - **Medyan (Median)**: Verinin ortanca değerini belirler, aşırı uç değerlerden etkilenmez.
    - **Standart Sapma (Standard Deviation)**: Verinin ne kadar yayıldığını gösterir, yüksek değerler verinin geniş bir aralığa yayıldığını belirtir.
    - **Minimum ve Maksimum (Min, Max)**: Verinin en küçük ve en büyük değerlerini gösterir.

### Korelasyon Analizi
Özellikler arasındaki korelasyonu analiz edin ve yüksek korelasyonlu özellikleri belirleyin.

1. **Korelasyon Matrisi**: Özellikler arasındaki korelasyonu ölçmek için korelasyon matrisi oluşturun. Korelasyon, iki değişken arasındaki ilişkinin gücünü ve yönünü gösterir. Pozitif korelasyon (+1) iki değişkenin birlikte arttığını, negatif korelasyon (-1) ise bir değişken artarken diğerinin azaldığını gösterir.

    - **Korelasyon Katsayısı (Correlation Coefficient)**: İki değişken arasındaki doğrusal ilişkinin gücünü ve yönünü ölçer. Değerler -1 ile +1 arasında değişir.
        - **+1**: Mükemmel pozitif korelasyon
        - **0**: Hiçbir korelasyon yok
        - **-1**: Mükemmel negatif korelasyon

### Veri Görselleştirme
Müşteri kaybı ve diğer özellikler arasındaki ilişkileri görselleştirin.

1. **Histogramlar**: Her bir özellik için veri dağılımını görselleştirin. Histogramlar, verinin nasıl dağıldığını ve hangi değerlerin daha sık tekrarlandığını gösterir.
2. **Kutu Grafikler (Box Plots)**: Verinin merkezi eğilimlerini ve yayılımını görselleştirir. Aşırı uç (outlier) değerleri tespit etmek için kullanılır.
3. **Dağılım Grafikler (Scatter Plots)**: İki değişken arasındaki ilişkiyi görselleştirir. Değişkenlerin nasıl bir ilişki içinde olduğunu görmek için kullanılır.
4. **Isı Haritaları (Heatmaps)**: Korelasyon matrisini görselleştirmek için kullanılır. Yüksek korelasyonlu özellikleri hızlıca belirlemeye yardımcı olur.
5. **Çubuk Grafikler (Bar Plots)**: Kategorik verilerin dağılımını ve sayısını görselleştirir. Özellikle müşteri kaybı gibi ikili değişkenler için kullanışlıdır.

Bu adımlar, veri setinin yapısını ve ana özelliklerini daha iyi anlamak için gerekli analizleri ve görselleştirmeleri kapsar. Keşifsel veri analizi, modelleme sürecinde hangi özelliklerin daha önemli olduğunu belirlemeye yardımcı olur ve veri setindeki olası sorunları (örneğin, aşırı uç değerler) tespit etmeyi sağlar.

## Özellik Mühendisliği

Özellik mühendisliği, veri setindeki ham verileri daha anlamlı ve modelleme için daha uygun hale getirmek amacıyla dönüştürme ve yeni özellikler oluşturma sürecidir. Bu adım, modelin performansını artırmak için kritik öneme sahiptir. Aşağıda, Telco Customer Churn veri seti için yapılması gereken özellik mühendisliği adımları detaylandırılmıştır.

### Yeni Özellikler Oluşturma
Veriyi daha anlamlı hale getirmek için yeni özellikler oluşturun.

1. **Mevcut Verilerden Türetilmiş Yeni Özellikler**: Mevcut sütunlardan daha fazla bilgi içeren yeni özellikler türetebilirsiniz. Bu, veriyi daha zengin ve modelleme için daha uygun hale getirir. Örneğin:
   - **Müşteri Yaşı (Customer Age)**: "tenure" ve müşterinin kaydolma tarihinden hesaplanabilir.
   - **Aylık Ortalama Harcama (Average Monthly Spend)**: "TotalCharges" ve "tenure" kullanılarak hesaplanabilir.
   - **Aile Durumu (Family Status)**: "Partner" ve "Dependents" kullanılarak müşterinin aile durumu belirlenebilir.

### Özellik Dönüşümü
Kategorik değişkenleri sayısal değerlere dönüştürün ve gerektiğinde özellik ölçeklendirmesi yapın.

1. **Kategorik Değişkenleri Sayısal Değerlere Dönüştürme (Encoding)**: Kategorik verileri modelin anlayabileceği sayısal formatlara dönüştürmek gereklidir. Aşağıdaki yöntemler kullanılabilir:
   - **Etiket Kodlama (Label Encoding)**: Her kategoriye benzersiz bir sayısal değer atanır. Bu yöntem, kategoriler arasında doğal bir sıralama olduğunda kullanışlıdır.
   - **Tek Sıcak Kodlama (One-Hot Encoding)**: Her kategori için ayrı bir sütun oluşturulur ve ilgili kategoriye sahip gözlemler için bu sütunda 1, diğerlerinde 0 değeri atanır. Bu yöntem, kategoriler arasında sıralama olmadığında kullanışlıdır.

2. **Özellik Ölçeklendirme (Feature Scaling)**: Özelliklerin farklı ölçeklerde olması, modelin performansını olumsuz etkileyebilir. Bu nedenle, özellikleri benzer ölçeklere getirmek önemlidir. Aşağıdaki yöntemler kullanılabilir:
   - **Min-Max Ölçeklendirme**: Her bir özellik, minimum ve maksimum değerleri arasında yeniden ölçeklendirilir, genellikle [0, 1] aralığında.
   - **Z-Puanı Standardizasyonu (Z-Score Standardization)**: Her özellik, ortalaması 0 ve standart sapması 1 olacak şekilde ölçeklendirilir. Bu yöntem, verinin normal dağılıma yakın olduğu durumlarda etkilidir.

### Özellik Mühendisliği Adımları
Aşağıda, özellik mühendisliği sürecinde yapılacak işlemler adım adım özetlenmiştir:

1. **Mevcut Verilerden Yeni Özellikler Türetme**:
   - Müşterilerin mevcut verilerinden daha anlamlı ve modelin performansını artıracak yeni özellikler oluşturun.
   - Bu yeni özellikler, müşteri davranışlarını ve özelliklerini daha iyi temsil edebilir.

2. **Kategorik Verileri Dönüştürme**:
   - Kategorik değişkenleri sayısal değerlere dönüştürmek için etiket kodlama veya tek sıcak kodlama yöntemlerini kullanın.
   - Bu dönüşümler, makine öğrenimi algoritmalarının kategorik verilerle daha iyi çalışmasını sağlar.

3. **Özellik Ölçeklendirme**:
   - Özelliklerin farklı ölçeklerde olmasının modelin performansını olumsuz etkilememesi için min-max ölçeklendirme veya z-puanı standardizasyonu gibi yöntemleri kullanarak özellikleri benzer ölçeklere getirin.
   - Ölçeklendirme, özellikle mesafe tabanlı algoritmalar (örneğin, K-En Yakın Komşu) ve gradient tabanlı algoritmalar (örneğin, Lojistik Regresyon) için önemlidir.

Bu adımlar, veri setini daha anlamlı ve modelleme için daha uygun hale getirecektir. Özellik mühendisliği, modelin genel performansını artırmak ve daha doğru tahminler yapmak için kritik bir süreçtir.

## Modelleme

Modelleme süreci, veriyi eğitim ve test setlerine ayırma, uygun modelleri seçme, modelleri eğitme ve test etme adımlarını içerir. Bu süreç, veri bilimi projelerinin temel aşamalarından biridir ve doğru şekilde uygulanması modelin performansını ve doğruluğunu etkiler. Aşağıda, Telco Customer Churn veri seti için modelleme adımları detaylandırılmıştır.

### Veri Bölme
Veriyi eğitim ve test setlerine ayırın.

1. **Eğitim ve Test Verisi Oranlarını Belirleme**: Veriyi eğitim ve test setlerine ayırırken genellikle %80 eğitim ve %20 test oranı kullanılır. Bu oranlar, modelin performansını değerlendirmek için yeterli veri sağlar.
   - **Eğitim Seti**: Modelin öğrenmesi için kullanılan veri seti.
   - **Test Seti**: Modelin performansını değerlendirmek için kullanılan veri seti.

### Model Seçimi
Müşteri kaybını tahmin etmek için uygun modelleri seçin.

1. **Lojistik Regresyon (Logistic Regression)**: İkili sınıflandırma problemleri için yaygın olarak kullanılan bir modeldir. Müşteri kaybını (churn) tahmin etmek için uygundur.
   - **Avantajları**: Basit, hızlı ve yorumlanabilir.
   - **Dezavantajları**: Doğrusal ilişkileri iyi modelleyemez.
   
2. **Karar Ağaçları (Decision Trees)**: Veriyi ağaç yapısı şeklinde bölerek sınıflandırma yapar.
   - **Avantajları**: Kolay yorumlanabilir, kategorik ve sayısal verilerle iyi çalışır.
   - **Dezavantajları**: Aşırı uyum (overfitting) riski taşır.
   
3. **Random Forest**: Birden fazla karar ağacının birleşimidir. Karar ağaçlarındaki aşırı uyumu azaltır.
   - **Avantajları**: Genellikle yüksek doğruluk sağlar, aşırı uyuma karşı dirençlidir.
   - **Dezavantajları**: Hesaplama maliyeti yüksektir.

### Model Eğitimi
Seçilen modelleri eğitim verisi üzerinde eğitin.

1. **Model Eğitimi**: Eğitim setini kullanarak modelleri eğitin. Bu adımda model, verideki desenleri ve ilişkileri öğrenir.
   - **Hiperparametre Ayarları**: Modelin performansını optimize etmek için hiperparametreler ayarlanır. Örneğin, karar ağaçlarında ağaç derinliği, random forest'ta ağaç sayısı gibi.

### Değerlendirme
Modelleri test verisi üzerinde test edin ve performanslarını değerlendirin.

1. **Model Performansını Değerlendirme**: Test setini kullanarak modelin performansını değerlendirin. Performans değerlendirmesi için çeşitli metrikler kullanılır:
   - **Doğruluk (Accuracy)**: Doğru tahminlerin toplam tahminlere oranı.
   - **Hassasiyet (Precision)**: Doğru pozitif tahminlerin toplam pozitif tahminlere oranı.
   - **Geri Çağırma (Recall)**: Doğru pozitif tahminlerin gerçek pozitif durumlara oranı.
   - **F1 Skoru**: Hassasiyet ve geri çağırmanın harmonik ortalaması.

2. **Modelin Test Edilmesi**: Modelin test verisi üzerindeki tahminlerini değerlendirerek gerçek performansını ölçün. Bu adım, modelin genelleme yeteneğini gösterir ve aşırı uyum (overfitting) olup olmadığını kontrol eder.

Bu adımlar, veri setini modellemek ve müşteri kaybını tahmin etmek için gerekli işlemleri içerir. Modelleme süreci, doğru modelin seçilmesi, eğitilmesi ve performansının değerlendirilmesi ile tamamlanır. Doğru uygulanmış bir modelleme süreci, güvenilir ve doğru tahminler elde etmenizi sağlar.




