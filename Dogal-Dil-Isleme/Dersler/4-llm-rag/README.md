<<<<<<< HEAD
# 🤖 Yapay Zeka Kulübü – Streamlit RAG Chatbot

Bu proje, **Streamlit** tabanlı, **Groq LLM API** kullanan ve **RAG (Retrieval-Augmented Generation)** mimarisiyle çalışan bir sohbet uygulamasıdır.

Kullanıcı:

* Çoklu sohbet (chat history)
* Dosya yükleme (PDF / TXT / DOCX)
* Dosya içeriğine dayalı soru-cevap
  özelliklerini kullanabilir.

---

## 🚀 Özellikler

* 📂 PDF / TXT / DOCX dosya yükleme
* 💬 Çoklu sohbet yönetimi
* 🧠 Dosya içeriğine dayalı akıllı cevaplar (RAG)
* ⚡ Groq (LLaMA tabanlı) hızlı LLM entegrasyonu
* 🎨 Özelleştirilmiş Streamlit arayüzü

---

## 🧠 RAG (Retrieval-Augmented Generation) Mimarisi

Bu bölüm iki parçadan oluşur:

1. **Bu projede kullanılan RAG yaklaşımı**
2. **Standart (klasik) RAG mimarisi** ve öğrenilmesi gereken kritik noktalar

---

## 1️⃣ Bu Projedeki RAG Mimarisi (Lightweight / Heuristic RAG)

Bu projede **Vector Database kullanılmadan**, hafif ama etkili bir RAG yaklaşımı uygulanmıştır.
Amaç: **LLM’e tüm dosyayı vermek yerine, soruyla en alakalı bölümü vermek**.

### 📂 Dosya İşleme

* Yüklenen dosya (PDF / TXT / DOCX) **ham metne** çevrilir
* Metin `\n\n` kullanılarak **paragraflara bölünür**
* Çok kısa ve anlamsız paragraflar elenir

### 🔍 Retrieval (Bilgi Getirme Mantığı)

Kullanıcı soru sorduğunda:

1. Soru **küçük harfe çevrilir** ve kelimelere ayrılır
2. Her paragraf için şu skor hesaplanır:

```python
score = sum(1 for keyword in keywords if keyword in para_lower)
```

Yani:

* Soru kelimeleri paragraf içinde geçiyorsa skor artar
* Skoru 0 olan paragraflar elenir

### 🧠 En İlgili Bağlamın Seçilmesi

* Paragraflar **skora göre sıralanır**
* En iyi **ilk 5 paragraf** alınır
* Toplam bağlam **maksimum 2000 karakterle sınırlandırılır**

Eğer hiçbir eşleşme yoksa:

* Dosyanın **ilk kısmı** fallback olarak kullanılır

### 🧩 Augmentation (Prompt Zenginleştirme)

Bulunan bağlam, **system prompt** içine eklenir:

> "Aşağıdaki dosya içeriğine dayanarak cevap ver"

Bu sayede LLM:

* Dosya dışına çıkmaz
* Halüsinasyon (uydurma bilgi) ihtimali azalır
* Daha kontrollü ve tutarlı cevap verir

### ✨ Generation (Cevap Üretimi)

* Groq LLM kullanılır
* Son **3 mesaj** bağlama eklenir
* Her mesaj **400 karakterle sınırlandırılır**
* `max_tokens = 2048`

Bu yaklaşım:

* Context taşmasını önler
* Performansı artırır

---

## 2️⃣ Standart (Klasik) RAG Mimarisi

Klasik RAG mimarisi **3 ana adımdan** oluşur:

```
User Query
   ↓
Embedding (Query)
   ↓
Vector DB (Similarity Search)
   ↓
Relevant Documents
   ↓
Prompt + Context
   ↓
LLM Response
```

### 🧱 1. Document Indexing (Offline Aşama)

Bu aşama **önceden** yapılır:

* Dokümanlar parçalara bölünür (chunking)
* Her parça embedding’e çevrilir
* Vector Database’e kaydedilir

Önemli parametreler:

* `chunk_size`
* `chunk_overlap`
* embedding modeli

📌 Yanlış chunk ayarı → kötü retrieval

---

### 🔍 2. Retrieval (Online Aşama)

Kullanıcı soru sorduğunda:

* Soru embedding’e çevrilir
* Vector DB’de **semantic similarity search** yapılır
* En benzer `top-k` parça seçilir

Önemli kavramlar:

* cosine similarity
* top-k
* metadata filtering

📌 Bu aşama RAG’in **en kritik kısmıdır**

---

### 🧠 3. Generation

* Seçilen dokümanlar prompt’a eklenir
* LLM sadece bu bağlama dayanarak cevap verir

Önemli noktalar:

* Context length limiti
* Prompt engineering
* Kaynak dışına çıkmama (grounding)

---

## 🎯 RAG Öğrenirken Bilinmesi Gereken En Önemli Konular

### ✅ Mutlaka Öğrenilmesi Gerekenler

* Chunking stratejileri
* Embedding nedir, nasıl çalışır
* Vector similarity (cosine, dot-product)
* Context window / token limiti
* Hallucination neden olur

### ⚠️ En Sık Yapılan Hatalar

* Tüm dokümanı prompt’a koymak
* Çok büyük chunk kullanmak
* Retrieval kalitesini test etmemek
* Prompt’u kontrolsüz bırakmak

---

## 🆚 Bu Proje vs Klasik RAG

| Özellik    | Bu Proje                    | Klasik RAG               |
| ---------- | --------------------------- | ------------------------ |
| Vector DB  | ❌                           | ✅                        |
| Embedding  | ❌                           | ✅                        |
| Kurulum    | Çok Kolay                   | Orta / Zor               |
| Performans | Küçük dosya için iyi        | Büyük veri için mükemmel |
| Öğrenme    | Yeni başlayanlar için ideal | Production-ready         |

---

## 📌 Sonuç

Bu proje:

* RAG mantığını **basit ve anlaşılır** şekilde öğretir
* Streamlit + LLM entegrasyonunu gösterir
* Klasik RAG’e geçiş için sağlam bir temel oluşturur

🚀
=======
# 🤖 Yapay Zeka Kulübü – Streamlit RAG Chatbot

Bu proje, **Streamlit** tabanlı, **Groq LLM API** kullanan ve **RAG (Retrieval-Augmented Generation)** mimarisiyle çalışan bir sohbet uygulamasıdır.

Kullanıcı:

* Çoklu sohbet (chat history)
* Dosya yükleme (PDF / TXT / DOCX)
* Dosya içeriğine dayalı soru-cevap
  özelliklerini kullanabilir.

---

## 🚀 Özellikler

* 📂 PDF / TXT / DOCX dosya yükleme
* 💬 Çoklu sohbet yönetimi
* 🧠 Dosya içeriğine dayalı akıllı cevaplar (RAG)
* ⚡ Groq (LLaMA tabanlı) hızlı LLM entegrasyonu
* 🎨 Özelleştirilmiş Streamlit arayüzü

---

## 🧠 RAG (Retrieval-Augmented Generation) Mimarisi

Bu bölüm iki parçadan oluşur:

1. **Bu projede kullanılan RAG yaklaşımı**
2. **Standart (klasik) RAG mimarisi** ve öğrenilmesi gereken kritik noktalar

---

## 1️⃣ Bu Projedeki RAG Mimarisi (Lightweight / Heuristic RAG)

Bu projede **Vector Database kullanılmadan**, hafif ama etkili bir RAG yaklaşımı uygulanmıştır.
Amaç: **LLM’e tüm dosyayı vermek yerine, soruyla en alakalı bölümü vermek**.

### 📂 Dosya İşleme

* Yüklenen dosya (PDF / TXT / DOCX) **ham metne** çevrilir
* Metin `\n\n` kullanılarak **paragraflara bölünür**
* Çok kısa ve anlamsız paragraflar elenir

### 🔍 Retrieval (Bilgi Getirme Mantığı)

Kullanıcı soru sorduğunda:

1. Soru **küçük harfe çevrilir** ve kelimelere ayrılır
2. Her paragraf için şu skor hesaplanır:

```python
score = sum(1 for keyword in keywords if keyword in para_lower)
```

Yani:

* Soru kelimeleri paragraf içinde geçiyorsa skor artar
* Skoru 0 olan paragraflar elenir

### 🧠 En İlgili Bağlamın Seçilmesi

* Paragraflar **skora göre sıralanır**
* En iyi **ilk 5 paragraf** alınır
* Toplam bağlam **maksimum 2000 karakterle sınırlandırılır**

Eğer hiçbir eşleşme yoksa:

* Dosyanın **ilk kısmı** fallback olarak kullanılır

### 🧩 Augmentation (Prompt Zenginleştirme)

Bulunan bağlam, **system prompt** içine eklenir:

> "Aşağıdaki dosya içeriğine dayanarak cevap ver"

Bu sayede LLM:

* Dosya dışına çıkmaz
* Halüsinasyon (uydurma bilgi) ihtimali azalır
* Daha kontrollü ve tutarlı cevap verir

### ✨ Generation (Cevap Üretimi)

* Groq LLM kullanılır
* Son **3 mesaj** bağlama eklenir
* Her mesaj **400 karakterle sınırlandırılır**
* `max_tokens = 2048`

Bu yaklaşım:

* Context taşmasını önler
* Performansı artırır

---

## 2️⃣ Standart (Klasik) RAG Mimarisi

Klasik RAG mimarisi **3 ana adımdan** oluşur:

```
User Query
   ↓
Embedding (Query)
   ↓
Vector DB (Similarity Search)
   ↓
Relevant Documents
   ↓
Prompt + Context
   ↓
LLM Response
```

### 🧱 1. Document Indexing (Offline Aşama)

Bu aşama **önceden** yapılır:

* Dokümanlar parçalara bölünür (chunking)
* Her parça embedding’e çevrilir
* Vector Database’e kaydedilir

Önemli parametreler:

* `chunk_size`
* `chunk_overlap`
* embedding modeli

📌 Yanlış chunk ayarı → kötü retrieval

---

### 🔍 2. Retrieval (Online Aşama)

Kullanıcı soru sorduğunda:

* Soru embedding’e çevrilir
* Vector DB’de **semantic similarity search** yapılır
* En benzer `top-k` parça seçilir

Önemli kavramlar:

* cosine similarity
* top-k
* metadata filtering

📌 Bu aşama RAG’in **en kritik kısmıdır**

---

### 🧠 3. Generation

* Seçilen dokümanlar prompt’a eklenir
* LLM sadece bu bağlama dayanarak cevap verir

Önemli noktalar:

* Context length limiti
* Prompt engineering
* Kaynak dışına çıkmama (grounding)

---

## 🎯 RAG Öğrenirken Bilinmesi Gereken En Önemli Konular

### ✅ Mutlaka Öğrenilmesi Gerekenler

* Chunking stratejileri
* Embedding nedir, nasıl çalışır
* Vector similarity (cosine, dot-product)
* Context window / token limiti
* Hallucination neden olur

### ⚠️ En Sık Yapılan Hatalar

* Tüm dokümanı prompt’a koymak
* Çok büyük chunk kullanmak
* Retrieval kalitesini test etmemek
* Prompt’u kontrolsüz bırakmak

---

## 🆚 Bu Proje vs Klasik RAG

| Özellik    | Bu Proje                    | Klasik RAG               |
| ---------- | --------------------------- | ------------------------ |
| Vector DB  | ❌                           | ✅                        |
| Embedding  | ❌                           | ✅                        |
| Kurulum    | Çok Kolay                   | Orta / Zor               |
| Performans | Küçük dosya için iyi        | Büyük veri için mükemmel |
| Öğrenme    | Yeni başlayanlar için ideal | Production-ready         |

---

## 📌 Sonuç

Bu proje:

* RAG mantığını **basit ve anlaşılır** şekilde öğretir
* Streamlit + LLM entegrasyonunu gösterir
* Klasik RAG’e geçiş için sağlam bir temel oluşturur

🚀
>>>>>>> a815bb6 (Hafta 4 - IMDb Sentiment Analysis ödevi)
