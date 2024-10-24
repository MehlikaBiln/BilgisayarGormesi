import os
import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin  # URL birleştirme için gerekli

# YOLO dosyalarının yolları
yolo_classes = r"C:\Users\MONSTER\PycharmProjects\pythonProject\models\coco.names"
yolo_weights = r"C:\Users\MONSTER\PycharmProjects\pythonProject\models\yolov4.weights"
yolo_config = r"C:\Users\MONSTER\PycharmProjects\pythonProject\models\yolov4.cfg"

# Net'i yükle
try:
    net = cv2.dnn.readNet(yolo_weights, yolo_config)
    print("Model başarıyla yüklendi.")
except cv2.error as e:
    print(f"Hata: {e}")

# Sınıf isimlerini okuma
with open(yolo_classes, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# İnsan sınıfının indexi
human_class_index = classes.index("person")

# URL'lerin bulunduğu dosya veya listeyi okuma
url_file = "dosya.txt"  # URLs.txt dosyasının yolu
with open(url_file, "r") as f:
    urls = [line.strip().replace('"', '') for line in f.readlines()]  # Tırnak işaretlerini kaldırma

# Sonuçları tutmak için bir liste
results = []

for url in urls:
    if not url.startswith("http"):  # Geçerli bir URL olup olmadığını kontrol et
        print(f"Görsel bulunamadı: {url}")
        results.append((url, "Geçersiz URL"))
        continue

    # URL'den sayfayı indirme
    try:
        response = requests.get(url)
        html_content = response.text

        # BeautifulSoup ile HTML'i parse etme
        soup = BeautifulSoup(html_content, "html.parser")

        # Resimleri bulma
        img_tags = soup.find_all("img")  # Tüm <img> etiketlerini bul
        img_urls = [img['src'] for img in img_tags if 'src' in img.attrs]

        # İlk resim URL'sini al
        if img_urls:
            image_url = img_urls[0]  # İlk görsel URL'sini kullan
            print(f"Görsel URL: {image_url}")

            # Göreceli URL'yi tam URL'ye dönüştürme
            full_image_url = urljoin(url, image_url)

            # Resmi indirme
            image_response = requests.get(full_image_url)
            image = np.asarray(bytearray(image_response.content), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            # Görüntü yükleme kontrolü
            if image is None:
                print(f"Görüntü yüklenemedi: {url}")
                results.append((url, "İndirilemedi"))
                continue

        else:
            print(f"Görsel bulunamadı: {url}")
            results.append((url, "Görsel bulunamadı"))
            continue

    except Exception as e:
        print(f"Görüntü indirilemedi: {url}, Hata: {e}")
        results.append((url, "İndirilemedi"))
        continue

    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    outputs = net.forward(output_layers)

    person_detected = False
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)


                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box

            if class_ids[i] == human_class_index:
                person_detected = True
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Yeşil kutu
                cv2.putText(image, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if person_detected:
        results.append((url, "İnsan"))
    else:
        results.append((url, "İnsan Değil"))

    # Resmi ekranda gösterme
    cv2.imshow(f"Sonuç: {url}", image)
    cv2.waitKey(0)  # Resmin kapanması için bir tuşa basılmasını bekle
    cv2.destroyAllWindows()

# Sonuçları yazdırma
for url, result in results:
    print(f"{url}: {result}")
