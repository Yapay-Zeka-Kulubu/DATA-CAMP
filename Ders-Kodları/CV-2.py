import cv2
import numpy as np

def perspektif_donusum(frame):

    yukseklik, genislik = frame.shape[:2]
    
    kaynak_noktalar = np.float32([
        [580, 460],   # Sol üst
        [1340, 460],  # Sağ üst
        [200, 1080],  # Sol alt
        [1720, 1080]  # Sağ alt
    ])

    hedef_noktalar = np.float32([
        [200, 0],      # Sol üst
        [1720, 0],     # Sağ üst
        [200, 1080],   # Sol alt
        [1720, 1080]   # Sağ alt
    ])
    
    matris = cv2.getPerspectiveTransform(kaynak_noktalar, hedef_noktalar)
    ters_matris = cv2.getPerspectiveTransform(hedef_noktalar, kaynak_noktalar)
    
    donusturulmus = cv2.warpPerspective(frame, matris, (genislik, yukseklik))

    kirpilmis = donusturulmus[:, 200:1720]
    
    return kirpilmis, ters_matris, kaynak_noktalar

def gri_ve_threshold(frame, threshold_deger=180):
    gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gri_iyilestirilmis = clahe.apply(gri)
    
    _, threshold = cv2.threshold(gri_iyilestirilmis, threshold_deger, 255, cv2.THRESH_BINARY)
    
    return threshold

def morfolojik_islemler(binary_img, acma_iter=2, kapama_iter=2):
    kernel_dikey = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    kernel_kucuk = np.ones((3, 3), np.uint8)
    
    acma = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel_kucuk, iterations=acma_iter)
    kapama = cv2.morphologyEx(acma, cv2.MORPH_CLOSE, kernel_dikey, iterations=kapama_iter)
    
    return acma, kapama

def serit_cizgilerini_bul(binary_img):

    yukseklik, genislik = binary_img.shape
    
    alt_bolum = binary_img[yukseklik//2:, :]
    
    sutun_toplamlari = np.sum(alt_bolum, axis=0) / 255
    
    esik = np.max(sutun_toplamlari) * 0.3 if np.max(sutun_toplamlari) > 0 else 1

    beyaz_bolgeler = []
    baslangic = None
    
    for i in range(len(sutun_toplamlari)):
        if sutun_toplamlari[i] > esik:
            if baslangic is None:
                baslangic = i
        else:
            if baslangic is not None:
                merkez = (baslangic + i - 1) // 2
                beyaz_bolgeler.append(merkez)
                baslangic = None
    
    if baslangic is not None:
        merkez = (baslangic + len(sutun_toplamlari) - 1) // 2
        beyaz_bolgeler.append(merkez)
    
    if len(beyaz_bolgeler) > 1:
        filtrelenmis = [beyaz_bolgeler[0]]
        for bolge in beyaz_bolgeler[1:]:
            if bolge - filtrelenmis[-1] > 80:
                filtrelenmis.append(bolge)
        beyaz_bolgeler = filtrelenmis
    
    return beyaz_bolgeler

def cizgileri_ciz_perspektif(frame, serit_merkezleri):
    yukseklik, genislik = frame.shape[:2]
    
    sonuc = frame.copy()
    
    if len(serit_merkezleri) >= 2:
        en_sol = serit_merkezleri[0]
        en_sag = serit_merkezleri[-1]
        
        cv2.line(sonuc, (en_sol, 0), (en_sol, yukseklik), (255, 255, 255), 8)
        cv2.line(sonuc, (en_sag, 0), (en_sag, yukseklik), (255, 255, 255), 8)
        
        orta = (en_sol + en_sag) // 2
        
        cv2.line(sonuc, (orta, 0), (orta, yukseklik), (0, 0, 255), 12)
    
    return sonuc

def cizgileri_ciz_orijinal(frame, serit_merkezleri, ters_matris):
    yukseklik, genislik = frame.shape[:2]
    
    cizgi_frame = np.zeros_like(frame)
    
    if len(serit_merkezleri) >= 2:
        en_sol = serit_merkezleri[0] + 200
        en_sag = serit_merkezleri[-1] + 200
        
        cv2.line(cizgi_frame, (en_sol, 0), (en_sol, yukseklik), (255, 255, 255), 8)
        cv2.line(cizgi_frame, (en_sag, 0), (en_sag, yukseklik), (255, 255, 255), 8)
        
        orta = (en_sol + en_sag) // 2
        cv2.line(cizgi_frame, (orta, 0), (orta, yukseklik), (0, 0, 255), 12)
    
    orijinal_perspektif = cv2.warpPerspective(
        cizgi_frame, ters_matris, (genislik, yukseklik)
    )
    
    sonuc = cv2.addWeighted(frame, 1, orijinal_perspektif, 0.8, 0)
    
    return sonuc

def ana_program(video_yolu):
    cap = cv2.VideoCapture(video_yolu)
    
    if not cap.isOpened():
        print("Video açılamadı!")
        return
    
    pencere_adi = 'Serit Takip Sistemi'
    
    # Pencereyi normal modda aç ki kullanıcı boyutunu değiştirebilsin
    cv2.namedWindow(pencere_adi, cv2.WINDOW_NORMAL)
    
    cv2.createTrackbar('Threshold', pencere_adi, 180, 255, lambda x: None)
    cv2.createTrackbar('Acma', pencere_adi, 2, 10, lambda x: None)
    cv2.createTrackbar('Kapama', pencere_adi, 2, 10, lambda x: None)
    
    # Yeni: Gösterim ölçeği için trackbar (yüzde olarak)
    cv2.createTrackbar('Scale', pencere_adi, 50, 100, lambda x: None)  # varsayılan 50 yüzde
    
    print("=" * 60)
    print("🚗 ŞERİT TAKİP SİSTEMİ")
    print("=" * 60)
    print("Kontroller:")
    print("  • Threshold: 0-255")
    print("  • Acma: 0-10")
    print("  • Kapama: 0-10")
    print("  • Scale: 10-100 (görüntü boyutu)")
    print("  • Çıkış: 'q'")
    print("  • Duraklat: 'SPACE'")
    print("=" * 60)
    
    duraklat = False
    nihai = None
    
    while True:
        if not duraklat:
            ret, frame = cap.read()
            
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            threshold_deger = cv2.getTrackbarPos('Threshold', pencere_adi)
            acma_iter = cv2.getTrackbarPos('Acma', pencere_adi)
            kapama_iter = cv2.getTrackbarPos('Kapama', pencere_adi)

            orijinal = frame.copy()
            
            perspektif, ters_matris, kaynak_pts = perspektif_donusum(frame)
            threshold = gri_ve_threshold(perspektif, threshold_deger)
            acma, kapama = morfolojik_islemler(threshold, acma_iter, kapama_iter)

            serit_merkezleri = serit_cizgilerini_bul(kapama)
            perspektif_cizgili = cizgileri_ciz_perspektif(perspektif, serit_merkezleri)
            orijinal_cizgili = cizgileri_ciz_orijinal(
                orijinal, serit_merkezleri, ters_matris
            )

            threshold_renkli = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
            acma_renkli = cv2.cvtColor(acma, cv2.COLOR_GRAY2BGR)
            kapama_renkli = cv2.cvtColor(kapama, cv2.COLOR_GRAY2BGR)
            
            ust_uc = np.hstack([threshold_renkli, acma_renkli, kapama_renkli])
            
            h_ust = 360
            w_ust = int(ust_uc.shape[1] * h_ust / ust_uc.shape[0])
            ust_uc_kucuk = cv2.resize(ust_uc, (w_ust, h_ust))
            
            h_alt = 720
            w_alt = int(orijinal_cizgili.shape[1] * h_alt / orijinal_cizgili.shape[0])
            alt_buyuk = cv2.resize(orijinal_cizgili, (w_alt, h_alt))
            
            max_width = max(w_ust, w_alt)
            
            if w_ust < max_width:
                pad_ust = np.zeros((h_ust, max_width - w_ust, 3), dtype=np.uint8)
                ust_uc_kucuk = np.hstack([ust_uc_kucuk, pad_ust])
            
            if w_alt < max_width:
                pad_alt = np.zeros((h_alt, max_width - w_alt, 3), dtype=np.uint8)
                alt_buyuk = np.hstack([alt_buyuk, pad_alt])
            
            nihai = np.vstack([ust_uc_kucuk, alt_buyuk])
        
        if nihai is not None:
            # Ölçek trackbarını oku
            scale_yuzde = cv2.getTrackbarPos('Scale', pencere_adi)
            if scale_yuzde < 10:
                scale_yuzde = 10  # çok küçük olmasın
            scale = scale_yuzde / 100.0

            yeni_genislik = int(nihai.shape[1] * scale)
            yeni_yukseklik = int(nihai.shape[0] * scale)
            
            nihai_kucuk = cv2.resize(nihai, (720, 720))
            cv2.imshow(pencere_adi, nihai_kucuk)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            duraklat = not duraklat
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_dosyasi = r"1128.mp4"
    ana_program(video_dosyasi)
