# detect_people_yolo11n.py
# ---------------------------------------------------
# Demo phát hiện người trong ảnh bằng YOLO11n
# Cần cài: pip install ultralytics opencv-python matplotlib
# ---------------------------------------------------

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import sys

# ================== CONFIG ==================
CONF_THRESH = 0.5   # Ngưỡng confidence (0 → 1). Lớn hơn = ít false positive hơn
MODEL_PATH = "yolo11n.pt"   # model YOLO (n: nano, nhỏ gọn)
# ============================================

def detect_people(image_path):
    # 1. Load model YOLO11n
    model = YOLO(MODEL_PATH)   # sẽ tự tải model lần đầu

    # 2. Chạy dự đoán
    results = model(image_path)

    # 3. Hiển thị ảnh với bounding box
    for r in results:
        im_bgr = r.plot()  # ảnh BGR có vẽ box
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(im_rgb)
        plt.title("Kết quả phát hiện bằng YOLO11n")
        plt.axis("off")
        plt.show()

        # 4. Lọc ra đối tượng class = "person"
        people_boxes = []
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if names[cls_id] == "person" and conf >= CONF_THRESH:
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                people_boxes.append((xyxy, conf))

        print("Số người phát hiện:", len(people_boxes))
        for i, (bbox, conf) in enumerate(people_boxes, 1):
            print(f"Người {i}: BBox={bbox}, Conf={conf:.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách chạy: python detect_people_yolo11n.py <D:/Downloads/vd1.jpg>")
        sys.exit(1)

    image_path = sys.argv[1]
    detect_people(image_path)
