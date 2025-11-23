import cv2
import numpy as np
import matplotlib.pyplot as plt

def foto_okuma_temel(img_path):
    img = cv2.imread(img_path)
    
    if img is not None:
        print(f"Fotoğraf boyutu: {img.shape}")
        print(f"Veri tipi: {img.dtype}")
        print(f"Piksel sayısı: {img.size}")
        
        cv2.imshow('Orijinal Fotograf', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Fotoğraf okunamadı!")
    
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    cv2.imshow('Gri Tonlamali', img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite('kayit_edilen.jpg', img_gray)
    print("Fotoğraf kaydedildi!")


def foto_renk_donusumleri(img_path):
    img = cv2.imread(img_path)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title('RGB Fotograf')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap='gray')
    plt.title('Gri Tonlamali')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB))
    plt.title('HSV')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def video_okuma_temel(video_path):
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"FPS: {fps}")
    print(f"Boyut: {width}x{height}")
    print(f"Toplam Frame: {frame_count}")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Video bitti")
            break
        
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def video_kaydetme(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
    
    while True:
        ret, frame = cap.read()
        
        if ret:
            out.write(frame)
            cv2.imshow('Kayit Ediliyor...', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()



def matris_matematiksel_islemler(img_path, img_path2=None):
    img = cv2.imread(img_path)
    
    parlak = cv2.add(img, np.array([50.0]))
    karanlik = cv2.subtract(img, np.array([150,150,150]))
    kontrastli = cv2.multiply(img, np.array([1.5,1.5,1.5]))
    
    img_float = img.astype(np.float32)
    img_parlak = np.clip(img_float + 50, 0, 255).astype(np.uint8)
    
    if img_path2:
        img2 = cv2.imread(img_path2)
        if img2 is not None and img.shape == img2.shape:
            karisim = cv2.addWeighted(img, 0.7, img2, 0.3, 0)
            cv2.imshow('Karisim', karisim)
    
    cv2.imshow('Orijinal', img)
    cv2.imshow('Parlak', parlak)
    cv2.imshow('Karanlik', karanlik)
    cv2.imshow('Kontrastli', kontrastli)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def temel_filtreler(img_path):
    img = cv2.imread(img_path)
    
    blur = cv2.blur(img, (5, 5))
    gaussian = cv2.GaussianBlur(img, (27, 27), 0)
    median = cv2.medianBlur(img, 5)
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)
    
    plt.figure(figsize=(12, 10))
    
    images = [img, blur, gaussian, median, bilateral]
    titles = ['Orijinal', 'Ortalama Blur', 'Gaussian Blur', 
              'Median Blur', 'Bilateral Filter']
    
    for i in range(5):
        plt.subplot(2, 3, i+1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def kenar_bulma(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = cv2.magnitude(sobelx, sobely)
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    canny = cv2.Canny(gray, 100, 200)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Orijinal (Gri)')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(np.abs(sobelx), cmap='gray')
    plt.title('Sobel X')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(np.abs(sobely), cmap='gray')
    plt.title('Sobel Y')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(np.abs(sobel), cmap='gray')
    plt.title('Sobel Combined')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(np.abs(laplacian), cmap='gray')
    plt.title('Laplacian')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(canny, cmap='gray')
    plt.title('Canny')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def morfolojik_islemler(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 5), np.uint8)
    
    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    
    plt.figure(figsize=(12, 10))
    
    images = [binary, erosion, dilation, opening, closing, gradient]
    titles = ['Binary', 'Erosion', 'Dilation', 'Opening', 'Closing', 'Gradient']
    
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def geometrik_donusumler(img_path):
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    
    scaled_up = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    scaled_down = cv2.resize(img, (width//2, height//2), interpolation=cv2.INTER_AREA)
    
    center = (width//2, height//2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
    
    translation_matrix = np.float32([[1, 0, 100], [0, 1, 50]])
    translated = cv2.warpAffine(img, translation_matrix, (width, height))
    
    flip_horizontal = cv2.flip(img, 1)
    flip_vertical = cv2.flip(img, 0)
    flip_both = cv2.flip(img, -1)
    
    cv2.imshow('Orijinal', img)
    cv2.imshow('Buyutulmus', scaled_up)
    cv2.imshow('Donmus', rotated)
    cv2.imshow('Yatay Cevrilmis', flip_horizontal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def kontur_bulma(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Bulunan kontur sayısı: {len(contours)}")
    
    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        
        print(f"Alan: {area}")
        print(f"Çevre: {perimeter}")
    
    cv2.imshow('Konturlar', img_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def histogram_islemleri(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    equalized = cv2.equalizeHist(gray)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(2, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Orijinal')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.plot(hist)
    plt.title('Histogram')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')
    
    plt.subplot(2, 3, 3)
    plt.imshow(equalized, cmap='gray')
    plt.title('Histogram Eşitlenmiş')
    plt.axis('off')
    
    colors = ('b', 'g', 'r')
    plt.subplot(2, 3, 4)
    for i, color in enumerate(colors):
        hist_color = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist_color, color=color)
    plt.title('Renkli Histogram')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')
    
    plt.tight_layout()
    plt.show()


def renk_filtreleme(img_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    red_result = cv2.bitwise_and(img, img, mask=red_mask)
    green_result = cv2.bitwise_and(img, img, mask=green_mask)
    blue_result = cv2.bitwise_and(img, img, mask=blue_mask)
    
    cv2.imshow('Orijinal', img)
    cv2.imshow('Kirmizi Maske', red_mask)
    cv2.imshow('Kirmizi Filtre', red_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    IMAGE_PATH = 'f1.jpg'
    IMAGE_PATH2 = 'f2.png'
    VIDEO_PATH = 'v1.mkv'
    VIDEO_OUTPUT = 'kayit.avi'
    WEBCAM = 0
    
    #foto_okuma_temel(IMAGE_PATH)
    #foto_renk_donusumleri(IMAGE_PATH)
    
    #video_okuma_temel(VIDEO_PATH)
    #video_okuma_temel(WEBCAM)
    # video_kaydetme(WEBCAM, VIDEO_OUTPUT)
    

    #matris_matematiksel_islemler(IMAGE_PATH, IMAGE_PATH2)
    
    #temel_filtreler(IMAGE_PATH)
    #kenar_bulma(IMAGE_PATH)
    #morfolojik_islemler(IMAGE_PATH)
    
    #geometrik_donusumler(IMAGE_PATH)
  
    #kontur_bulma(IMAGE_PATH)
    # histogram_islemleri(IMAGE_PATH)
    # renk_filtreleme(IMAGE_PATH)
