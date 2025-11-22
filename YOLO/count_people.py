import cv2
import numpy as np
from yolo_pose import HumanDetection

# --- CẤU HÌNH ---
IMAGE_TO_PROCESS = '27.jpg'
CONFIDENCE_THRESHOLD = 0.1
IOU_THRESHOLD = 0.5         

def main():
    # 1. Khởi tạo model
    try:
        detector = HumanDetection(conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
        print("✅ Model phát hiện người đã sẵn sàng.")
    except Exception as e:
        print(f"Lỗi khi khởi tạo model: {e}")
        return

    # 2. Đọc ảnh
    frame = cv2.imread(IMAGE_TO_PROCESS)
    if frame is None:
        print(f"LỖI: Không thể đọc ảnh '{IMAGE_TO_PROCESS}'.")
        return

    print(f"🔎 Đang xử lý ảnh '{IMAGE_TO_PROCESS}'...")
    keypoints_data, boxes_data = detector.run_detection(source=frame)

    # 4. Đếm và vẽ kết quả
    person_count = 0
    if boxes_data is not None:
        person_count = len(boxes_data)
        print(f"\n--- KẾT QUẢ ---")
        print(f"👍 Đã phát hiện được: {person_count} người.")

        print("\n--- Tọa độ các Bounding Box sẽ được vẽ ---")
        for i, box in enumerate(boxes_data):
            # === SỬA LỖI Ở ĐÂY ===
            # Bỏ .cpu().numpy() vì box đã là numpy array
            print(f"Box #{i+1}: {box[:4]}") 
            
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Chỉ vẽ nếu box có kích thước hợp lệ
            if x2 > x1 and y2 > y1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print("-----------------------------------------")
    else:
        print("\n--- KẾT QUẢ ---")
        print("🤷 Không phát hiện được người nào trong ảnh.")

    # 5. Hiển thị kết quả
    cv2.putText(frame, f"So nguoi: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Ket Qua Dem Nguoi", frame)
    print("\nNhấn phím bất kỳ trên cửa sổ ảnh để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()