import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans

# ─── Hàm lấy polygon vùng áo dựa vào keypoints COCO:
def get_shirt_polygon(keypoints):
    # COCO keypoint indices: 5=left shoulder, 6=right shoulder, 11=left hip, 12=right hip
    pts = []
    for idx in (5, 6, 12, 11):
        x, y, conf = keypoints[idx]
        if conf < 0.3:  # confidence thấp thì bỏ
            return None
        pts.append((int(x), int(y)))
    return np.array(pts, dtype=np.int32)

# ─── Hàm cluster 5 màu + %:
def cluster_colors(pixels, n_colors=7):
    if pixels is None or len(pixels)==0:
        return [], []
    # chuyển sang float32 và normalize
    data = pixels.reshape(-1,3).astype(np.float32)
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(data)
    centers = kmeans.cluster_centers_.astype(int)  # màu BGR
    counts = np.bincount(kmeans.labels_, minlength=n_colors)
    percents = counts / counts.sum() * 100
    return centers, percents

# ─── Main
if __name__ == "__main__":
    model = YOLO("yolov8n-pose.pt")
    img   = cv2.imread("quyen.JPG")
    res   = model(img)[0]

    if len(res.boxes) == 0:
        raise RuntimeError("Không phát hiện ai")

    # Lấy keypoints dạng (17,3)
    kpts = res.keypoints.data[0].cpu().numpy()

    poly = get_shirt_polygon(kpts)
    if poly is None:
        print("Keypoints torso không đủ tin cậy.")
        exit()

    # Tạo mask và trích pixel vùng áo
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly, 1)
    shirt_pixels = img[mask==1]  # array (N,3)

    # Cluster về 5 màu
    centers, percents = cluster_colors(shirt_pixels, n_colors=5)

    # In kết quả
    print("5 màu chủ đạo (B G R) và phần trăm:")
    for (b,g,r), p in zip(centers, percents):
        print(f"  [{b:3d}, {g:3d}, {r:3d}]  → {p:.1f}%")

    # Vẽ kết quả lên ảnh demo
    sw = 40; sh = 40
    for i,(c,p) in enumerate(zip(centers,percents)):
        x0 = i*(sw+10)+10
        cv2.rectangle(img, (x0,10), (x0+sw,10+sh), tuple(int(x) for x in c), -1)
        cv2.putText(img, f"{p:.0f}%", (x0, 10+sh+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.polylines(img, [poly], True, (0,255,0), 2)
    cv2.imshow("Shirt Colors", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
