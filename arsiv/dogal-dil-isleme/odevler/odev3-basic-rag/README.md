# Doküman Tabanlı Soru–Cevap Sistemi  
**(Keyword Retrieval + Google Gemini API)**

Bu proje, bir metin dosyası (`document.txt`) üzerinden anahtar kelime tabanlı arama yaparak en alakalı paragrafları bulan ve yalnızca bu bağlam üzerinden Google Gemini API kullanarak cevap üreten basit ve kontrollü bir soru–cevap sistemidir.

Amaç, modelin **doküman dışına çıkmasını engellemek** ve yalnızca verilen içerik üzerinden cevap üretmesini sağlamaktır.

---

## Kurulum (Adım Adım)

### Python Ortamını Hazırlama

1. Bilgisayarınızda Python’un yüklü olup olmadığını kontrol edin:
   ```bash
   python --version
   ```
2. Python sürümünün **3.9 veya üzeri** olduğundan emin olun.
3. Eğer Python yüklü değilse veya sürüm eskiyse, Python’u güncelleyin.

---

### Gerekli Kütüphaneyi Yükleme

1. Terminali veya komut satırını açın.
2. Aşağıdaki komutu çalıştırın:
   ```bash
   pip install google-genai
   ```
3. Yüklemenin başarılı olduğunu doğrulamak için:
   ```bash
   pip show google-generativeai
   ```

---

### API Anahtarını Tanımlama

API anahtarı koda yazılmaz, **environment variable** olarak verilir.

1. Repo kökündeki `.env.example` dosyasını `.env` olarak kopyalayın.
2. `GEMINI_API_KEY` satırına kendi Google Gemini anahtarınızı yazın:
   ```
   GEMINI_API_KEY=buraya_kendi_anahtariniz
   ```
3. Alternatif olarak terminalden de verebilirsiniz:
   ```bash
   export GEMINI_API_KEY="buraya_kendi_anahtariniz"
   ```

> ⚠️ `.env` dosyası `.gitignore` içindedir; **asla** commit etmeyin.
> API anahtarını koda hard-code etmek, repo public olduğunda anahtarın sızması demektir.

---

## Uygulamayı Çalıştırma (Adım Adım)

1. Terminali proje kök dizininde açın (`main.py` dosyasının bulunduğu yer).
2. Aşağıdaki komutu çalıştırın:
   ```bash
   python main.py
   ```
3. Program çalıştığında ekranda şu istem görüntülenir:
   ```text
   Sorunuzu giriniz:
   ```
4. Sorunuzu yazın ve **Enter** tuşuna basın.
5. Sistem, doküman içeriğine göre cevap üretir veya bilgi yoksa uyarı verir.

---

## Çalışma Mantığı (Adım Adım)

1. `document.txt` dosyası okunur.
2. Metin satırlara/paragraflara bölünür.
3. Kullanıcının sorusu küçük harfe çevrilir.
4. Sorudan anahtar kelimeler çıkarılır.
5. Her paragraf içindeki kelimeler analiz edilir.
6. Anahtar kelime kesişimine göre paragraflar puanlanır.
7. En yüksek puanlı `top_k` paragraf seçilir.
8. Seçilen paragraflar LLM’e bağlam olarak verilir.
9. Model yalnızca bu bağlam üzerinden cevap üretir.
10. Uygun bilgi yoksa aşağıdaki mesaj döndürülür:
    ```text
    Bu bilgi dökümanda bulunmamaktadır.
    ```

---

## Güvenlik ve Kısıtlar

1. Model yalnızca verilen bağlamla çalışır.
2. Bağlam dışı bilgi üretmesi sistem mesajı ile engellenmiştir.
3. Halüsinasyon riskini azaltmak için cevap alanı kısıtlanmıştır.
4. Doküman dışı bilgi üretimi bilinçli olarak yasaklanmıştır.

---

## Özelleştirme (Adım Adım)

### `top_k` Değerini Değiştirme

1. Aşağıdaki satırı bulun:
   ```python
   retrieve_relevant_paragraphs(paragraphs, question, top_k=3)
   ```
2. `top_k` değerini artırarak veya azaltarak daha fazla ya da daha az paragraf kullanılmasını sağlayabilirsiniz.

---

### Gelişmiş Arama Yöntemleri

Bu projedeki anahtar kelime tabanlı yapı yerine aşağıdaki yöntemler entegre edilebilir:

- TF-IDF
- Cosine Similarity
- Embedding tabanlı arama (vector search)

---

## Kullanım Senaryoları

- Ders notları için soru–cevap sistemleri
- Kurum içi dokümantasyon botları
- Teknik döküman arama araçları
- Eğitim ve sınav destek uygulamaları

---

## Lisans

Bu proje eğitim ve kişisel kullanım amaçlıdır.  
Ticari kullanım durumunda Google Gemini API’nin lisans ve kullanım koşulları dikkate alınmalıdır.

---

## Not

Bu yapı, basit ama kontrollü bir LLM kullanımını hedefler.  
Öncelik **doğru bağlam**, **kontrollü cevap** ve **öngörülebilir davranış**tır.
