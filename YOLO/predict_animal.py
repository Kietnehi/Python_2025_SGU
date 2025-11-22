import tensorflow as tf
import numpy as np
import cv2 # Sử dụng OpenCV để đọc ảnh
import os

# --- 1. CẤU HÌNH (BẠN CẦN CHỈNH SỬA Ở ĐÂY) ---

# Đường dẫn tới model .h5 bạn đã huấn luyện
MODEL_PATH = 'animal_classifier_model.h5'

# !!! QUAN TRỌNG: Đường dẫn tới ảnh bạn muốn dự đoán !!!
IMAGE_TO_PREDICT = 'pig.jpg' # <-- THAY ĐỔI ĐƯỜNG DẪN NÀY

# !!! QUAN TRỌNG: Đường dẫn tới thư mục dataset gốc (chứa 90 thư mục con) !!!
# Đây là đường dẫn bạn đã dùng trong file Train_animals.py
DATASET_PATH_FOR_LABELS = 'datasets/animals/animals' # <-- THAY ĐỔI ĐƯỜNG DẪN NÀY

# Các tham số cho hình ảnh (phải giống lúc train)
IMG_HEIGHT = 224
IMG_WIDTH = 224

# --- 2. CHUẨN BỊ NHÃN (CLASS LABELS) ---
# Đoạn code này sẽ đọc tên của 90 thư mục con để làm nhãn
try:
    class_names = sorted(os.listdir(DATASET_PATH_FOR_LABELS))
    if len(class_names) == 0:
        print(f"Lỗi: Không tìm thấy thư mục con nào trong '{DATASET_PATH_FOR_LABELS}'.")
        exit()
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy đường dẫn dataset '{DATASET_PATH_FOR_LABELS}'. Vui lòng cập nhật lại đường dẫn cho đúng.")
    exit()

print(f"✅ Đã tìm thấy {len(class_names)} lớp. 5 lớp đầu tiên: {class_names[:5]}...")

# --- 3. TẢI MODEL ĐÃ HUẤN LUYỆN ---
print("Đang tải model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Tải model thành công!")
except Exception as e:
    print(f"Lỗi khi tải model: {e}")
    exit()

# --- 4. HÀM TIỀN XỬ LÝ ẢNH VÀ DỰ ĐOÁN ---
def predict_animal(image_path, model_to_use, labels):
    """
    Hàm này nhận đường dẫn ảnh, model và danh sách nhãn,
    sau đó trả về tên con vật được dự đoán và độ tin cậy.
    """
    # a. Đọc và tiền xử lý ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: {image_path}")
        return None, None
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img_array = img_resized / 255.0
    
    # b. Mở rộng chiều để khớp với input của model
    img_batch = np.expand_dims(img_array, axis=0)
    
    # c. Thực hiện dự đoán
    predictions = model_to_use.predict(img_batch)
    
    # d. Xử lý kết quả
    predicted_index = np.argmax(predictions[0])
    predicted_label = labels[predicted_index]
    confidence = np.max(predictions[0]) * 100
    
    return predicted_label, confidence

# --- 5. CHẠY DỰ ĐOÁN ---
if os.path.exists(IMAGE_TO_PREDICT):
    predicted_animal, confidence_score = predict_animal(IMAGE_TO_PREDICT, model, class_names)
    
    if predicted_animal:
        print("\n--- KẾT QUẢ DỰ ĐOÁN ---")
        print(f"🐾 Loài vật: {predicted_animal.upper()}")
        print(f"🎯 Độ tin cậy: {confidence_score:.2f}%")
else:
    print(f"\n❌ Lỗi: Tệp ảnh '{IMAGE_TO_PREDICT}' không tồn tại. Vui lòng kiểm tra lại đường dẫn.")