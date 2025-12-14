# Metot Geliştirme Ödevi
## Amaç
Bu ödevin amacı, seçtiğiniz problem alanı için literatüre dayalı bir **metot geliştirme** önerisi üretmeniz ve bunu **sistematik bir rapor** formatında sunmanızdır. Rapor, yalnızca “ne yaptım” anlatımı değil, “neden böyle yaptım” gerekçelendirmesi içermelidir.

---

### Tablo A: Makale Kimliği ve Kapsam
| Alan | Bilgi |
|---|---|
| Makale başlığı |  |
| Yazarlar |  |
| Yıl |  |
| Yayın türü | Dergi / Konferans |
| İndeks bilgisi | SCI / SCI-E / Diğer |
| Problem tanımı |  |
| Temel katkı |  |
| Kod veya repo | Var / Yok (link) |

### Tablo B: Veri Setleri ve Protokol (Makale bazlı)
| Veri seti | Örnek sayısı | Sınıf sayısı | Bölme (train/val/test) | Modalite | Notlar |
|---|---:|---:|---|---|---|
|  |  |  |  |  |  |

### Tablo C: Veri Ön İşleme ve Artırma (Makale bazlı)
| Adım | Uygulama | Parametreler | Amaç | Not |
|---|---|---|---|---|
| Normalizasyon |  |  |  |  |
| Yeniden boyutlama |  |  |  |  |
| Augmentation |  |  |  |  |
| Etiket işleme |  |  |  |  |

### Tablo D: Model Mimarisi (Makale bazlı)
| Bileşen | Seçim | Detay | Gerekçe |
|---|---|---|---|
| Omurga (backbone) |  |  |  |
| Başlık (head) |  |  |  |
| Aktivasyonlar |  |  |  |
| Normalizasyon |  |  |  |
| Kayıp fonksiyonu |  |  |  |

### Tablo E: Eğitim Parametreleri (Makale bazlı)
| Parametre | Değer |
|---|---|
| Optimizer |  |
| Öğrenme oranı |  |
| LR scheduler |  |
| Batch size |  |
| Epoch |  |
| Weight decay |  |
| Erken durdurma |  |
| Donanım |  |
| Seed ve determinism |  |

### Tablo F: Sonuçlar ve Karşılaştırmalar (Makale bazlı)
| Veri seti | Metrikler | Sonuç | Baz çizgi | İyileşme | Not |
|---|---|---:|---:|---:|---|
|  | Accuracy / F1 / AUC / mAP vb |  |  |  |  |


## Kendi Önerdiğiniz Metot İçin Zorunlu Tablolar
Bu tablolar, “benim metot önerim” kısmında doldurulacaktır.

### Tablo H: Kullanılacak Veri Setleri Matrisi
| Veri seti | Kullanım amacı | Dahil mi | Dahil edilme gerekçesi | Risk / kısıt |
|---|---|---|---|---|
| Proje veri seti 1 | Train / Val / Test / External | Evet |  |  |
| Proje veri seti 2 |  | Evet |  |  |
| Ek veri seti (opsiyonel) |  |  |  |  |

### Tablo I: Önerilen Uçtan Uca Pipeline
| Aşama | Girdi | Çıktı | Yöntem | Parametreler |
|---|---|---|---|---|
| Veri alma |  |  |  |  |
| Temizleme |  |  |  |  |
| Ön işleme |  |  |  |  |
| Eğitim |  |  |  |  |
| Değerlendirme |  |  |  |  |
| Hata analizi |  |  |  |  |

### Tablo J: Önerilen Model ve Eğitim Konfigürasyonu
| Başlık | Öneri | Alternatifler | Seçim gerekçesi |
|---|---|---|---|
| Model omurgası |  |  |  |
| Head |  |  |  |
| Kayıp |  |  |  |
| Optimizer |  |  |  |
| LR schedule |  |  |  |
| Batch size |  |  |  |
| Epoch |  |  |  |
| Regularization |  |  |  |
| Augmentation |  |  |  |
| Seed stratejisi |  |  |  |



---
