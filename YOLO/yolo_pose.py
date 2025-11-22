# Nội dung mới cho file yolo_pose.py

from ultralytics import YOLO
import logging

# Cấu hình logging để giảm bớt các thông báo không cần thiết
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HumanDetection:
    """
    Lớp này được nâng cấp để có thể nhận các tham số conf và iou
    ngay từ khi khởi tạo.
    """
    def __init__(self, model_path='yolov8n-pose.pt', conf=0.25, iou=0.7):
        """
        Hàm khởi tạo model.

        Args:
            model_path (str): Đường dẫn tới file model .pt.
            conf (float): Ngưỡng tin cậy (confidence threshold).
            iou (float): Ngưỡng IoU (Intersection over Union).
        """
        logging.info(f"Khởi tạo Human Detection với model: {model_path}, conf={conf}, iou={iou}")
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        logging.info("Model đã được tải thành công.")

    def run_detection(self, source):
        """
        Chạy phát hiện trên một ảnh nguồn.

        Args:
            source: Ảnh đầu vào (định dạng numpy array mà OpenCV đọc được).

        Returns:
            Tuple: (keypoints_data, boxes_data)
                   keypoints_data: Dữ liệu keypoints cho mỗi người.
                   boxes_data: Dữ liệu bounding box cho mỗi người.
        """
        # Sử dụng các giá trị conf và iou đã được lưu lúc khởi tạo
        results = self.model.predict(source, conf=self.conf, iou=self.iou, verbose=False)

        if not results or len(results) == 0:
            return None, None

        result = results[0] # Chỉ lấy kết quả từ ảnh đầu tiên

        if result.keypoints is None or result.boxes is None:
            return None, None
            
        return result.keypoints.data.cpu().numpy(), result.boxes.data.cpu().numpy()