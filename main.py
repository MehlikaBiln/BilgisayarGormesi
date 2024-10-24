import cv2
import numpy as np

# Web kamerasını başlat
cap = cv2.VideoCapture(0)  # 0, varsayılan web kamerasını belirtir

if not cap.isOpened():
    print("Kamera açılamadı, lütfen bir kamera bağlayın.")
else:
    while True:
        # 1. Kameradan bir görüntü yakala
        ret, frame = cap.read()

        if not ret:
            print("Kameradan görüntü alınamadı, çıkılıyor...")
            break

        # 2. Gri tonlamaya çevir
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 3. Ortalama filtresi uygula
        avg_filtered_image = cv2.blur(gray, (5, 5))

        # 4. Laplace filtresi uygula
        laplace_filtered_image = cv2.Laplacian(avg_filtered_image, cv2.CV_64F)

        # 5. Kesin değerleri al ve normalleştir
        laplace_filtered_image_abs = np.abs(laplace_filtered_image)
        laplace_filtered_image_norm = cv2.normalize(laplace_filtered_image_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 6. Gri tonlama görüntülerini 3 kanala dönüştür
        avg_filtered_colored = cv2.cvtColor(avg_filtered_image, cv2.COLOR_GRAY2BGR)  # Ortalama filtreyi 3 kanala dönüştür
        laplace_filtered_colored = cv2.cvtColor(laplace_filtered_image_norm, cv2.COLOR_GRAY2BGR)  # Laplace filtresini 3 kanala dönüştür

        # 7. Başlıkları ekle
        cv2.putText(frame, 'Orijinal', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(avg_filtered_colored, 'Ortalama', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(laplace_filtered_colored, 'Laplace', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 8. Görüntüleri yeniden boyutlandır
        height, width = frame.shape[:2]
        new_width = int(width / 3)  # Her bir görüntünün yeni genişliği
        new_height = int(height * 0.5)  # Yüksekliği %50 azalt

        frame_resized = cv2.resize(frame, (new_width, new_height))
        avg_filtered_resized = cv2.resize(avg_filtered_colored, (new_width, new_height))
        laplace_filtered_resized = cv2.resize(laplace_filtered_colored, (new_width, new_height))

        # 9. Sonuçları yan yana göster (yan yana birleştir)
        combined_image = np.hstack((frame_resized, avg_filtered_resized, laplace_filtered_resized))

        # 10. Birleştirilmiş görüntüyü göster
        cv2.imshow('Orijinal - Ortalama - Laplace', combined_image)

        # 11. 'Esc' tuşuna basıldığında çık
        if cv2.waitKey(1) == 27:  # 27, Esc tuşunun ASCII kodudur
            print("Esc tuşuna basıldı, programdan çıkılıyor...")
            break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()