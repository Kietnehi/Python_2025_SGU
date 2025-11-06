# Python , Machine Learning , Deep Learning - SGU 2025

<div align="center">

| | | | |
|:--:|:--:|:--:|:--:|
| <img src="asset/MLjpg.jpg" width="180"/> | <img src="asset/DL.jpg" width="180"/> | <img src="asset/transformer.png" width="180"/> | <img src="asset/python.jpg" width="180"/> |

<br>

<img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python"/>
<img src="https://img.shields.io/badge/Deep%20Learning-PyTorch%20%7C%20TensorFlow-orange.svg" alt="Deep Learning"/>
<img src="https://img.shields.io/badge/Machine%20Learning-scikit--learn-green.svg" alt="Machine Learning"/>

</div>




## 📚 Giới thiệu

Repository này chứa các bài tập, dự án và tài liệu học tập về Python và Machine Learning/Deep Learning tại Đại học Sài Gòn (SGU) năm 2025. Nội dung bao gồm từ các kiến thức cơ bản về Python đến các mô hình Deep Learning hiện đại.

## 📋 Nội dung chính

### 🐍 Python Cơ bản
- **NumPy**: Xử lý mảng và tính toán số học
- **Pandas**: Phân tích và xử lý dữ liệu
- **Matplotlib**: Trực quan hóa dữ liệu
- **OOP (Object-Oriented Programming)**: Lập trình hướng đối tượng

### 🤖 Machine Learning
- **SVM (Support Vector Machine)**: Phân loại và hồi quy
- **Traditional ML Algorithms**: Các thuật toán học máy cổ điển
- **Feature Engineering**: Kỹ thuật tạo đặc trưng

### 🧠 Deep Learning - Computer Vision

#### Kiến trúc CNN (Convolutional Neural Networks)
- **VGG**: VGG16, VGG19 - Mạng tích chập sâu
- **ResNet**: Residual Networks - Mạng với skip connections
- **ViT (Vision Transformer)**: Transformer cho xử lý ảnh
- **CLIP**: Contrastive Language-Image Pre-training

#### Object Detection
- **YOLO**: You Only Look Once - Phát hiện đối tượng real-time
- Custom object detection applications

### 🔬 Deep Learning - Advanced

#### Sequence Models
- **Encoder-Decoder**: Kiến trúc cho sequence-to-sequence
- **Transformer**: Self-attention mechanism

#### Generative Models
- **GAN (Generative Adversarial Networks)**: Mạng đối sinh
- Ứng dụng tạo ảnh và data augmentation

#### Graph Neural Networks
- **GNN (Graph Neural Networks)**: Mạng nơ-ron đồ thị
- **GCN (Graph Convolutional Networks)**: Tích chập trên đồ thị

#### Unsupervised Learning
- **SOM (Self-Organizing Maps)**: Bản đồ tự tổ chức

## 📦 Dataset cho Project Python SGU

> **Lưu ý nhanh:** Đây là link Google Drive chứa **toàn bộ Dataset**. Bạn **phải tải nguyên folder về máy** rồi **chỉnh sửa lại đường dẫn (path) tới từng file dữ liệu** trong code trước khi chạy.

---

## 🔗 Dataset

## 📦 Dataset Download
<p align="center">
  <a href="https://drive.google.com/drive/folders/1DHVwFYHhsI0_yJycMtlq0ju0qLd5JQsv?usp=sharing" target="_blank">
    <img src="https://img.shields.io/badge/📂_Google_Drive-Dataset-blue?style=for-the-badge&logo=google-drive" alt="Dataset Google Drive">
  </a>
</p>

> ⚠️ **Quan trọng:**  
> Hãy **tải toàn bộ thư mục Dataset về máy** trước khi sử dụng.  
> Không nên chạy hoặc load dữ liệu trực tiếp từ Google Drive.


---
## 📁 Cấu trúc thư mục  (tree.txt)


```
Python project tree - 2025-11-06 09:48:46
Root: C:\Users\ADMIN\Desktop\python

├── asset
│   ├── DL.jpg
│   ├── GAN-model.pptx
│   ├── MLjpg.jpg
│   ├── python.jpg
│   └── transformer.png
├── CNN
│   ├── CNN_NhanDienMail_NhanDienAnh.ipynb
│   └── CNN_ViDuSo.ipynb
├── CODE_SVM
│   ├── [Solution]_Auto_Insurance_Prediction.ipynb
│   ├── [Solution]_Breast_Cancer_Recurrence_Classification.ipynb
│   ├── auto-insurance.csv
│   └── breast-cancer.csv
├── DecisionTree  - CNN
│   ├── cnn.ipynb
│   └── decisiontree.ipynb
├── EfficientNet
│   └── efficientnet.ipynb
├── Extended_on_Internet
│   ├── digit.png
│   ├── k_means.ipynb
│   ├── knn.ipynb
│   ├── notes.txt
│   └── svm.ipynb
├── img
│   └── dog_alaska.jpg
├── Nhập môn python
│   ├── BaiTapMonNhapMonPython.ipynb
│   ├── matplotlib.ipynb
│   ├── numpyy.ipynb
│   └── tuan3_pandas.ipynb
├── ResNet
│   ├── fine_tuned_resnet18_5cls_best.pth
│   └── resnet.ipynb
├── VGG
│   └── vgg.ipynb
├── README.md
├── requirements.txt
├── test.ipynb
└── tree.txt

(Excluded: __pycache__, venv, .venv, env, .git, .idea, .vscode, .mypy_cache, .pytest_cache, dist, build, .coverage, htmlcov)


```

## 🚀 Bắt đầu

### Yêu cầu hệ thống
```bash
Python 3.8+
pip install numpy pandas matplotlib
pip install scikit-learn
pip install torch torchvision  # PyTorch
pip install tensorflow keras   # TensorFlow
pip install ultralytics        # YOLO
```

### Cài đặt
```bash
# Clone repository
git clone https://github.com/Kietnehi/Python_2025_SGU.git
cd Python_2025_SGU

# Cài đặt dependencies
pip install -r requirements.txt
```

### Chạy Jupyter Notebook
```bash
jupyter notebook
```

## 📖 Tài liệu tham khảo

### Sách & Courses
- Deep Learning - Ian Goodfellow
- Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow
- CS231n: Convolutional Neural Networks for Visual Recognition
- CS224n: Natural Language Processing with Deep Learning

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer
- [Deep Residual Learning](https://arxiv.org/abs/1512.03385) - ResNet
- [YOLO: Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- [Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020) - CLIP
- [VGG](https://arxiv.org/abs/1409.1556) - VGG



## 👨‍💻 Tác giả

**Kietnehi**
- GitHub: [@Kietnehi](https://github.com/Kietnehi)
- Repository: [Python_2025_SGU](https://github.com/Kietnehi/Python_2025_SGU)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Hãy tạo Pull Request hoặc mở Issue nếu bạn có đề xuất cải tiến.

---

<div align="center">
  
  **⭐ Đừng quên star repo nếu bạn thấy hữu ích! ⭐**
  
  Made with ❤️ by SGU Students
  
</div>

## 🎓 **Liên kết học tập (Canvas & Projects)**

| 🧩 **Chủ đề** | 🔗 **Link Canva / Tài liệu học tập** |
|:--|:--|
| 🧮 **Numpy** | [Xem Canvas](https://drive.google.com/file/d/15mp6EOB68LNWKGEomKHZSMvcYsd4IUwr/view?usp=sharing) |
| 📊 **Matplotlib** | [Xem Canvas](https://www.canva.com/design/DAGZWHSgnCI/nZZyRT0o7w2OSxDM8Ys3Vw/edit) |
| 🧾 **Pandas** | [Xem Canvas](https://www.canva.com/design/DAGzVtrx2bc/cf3JB0ldvt2c4wEUGMFkUw/view) |
| ⚙️ **SVM (Support Vector Machine)** | [Xem Canvas](https://www.canva.com/design/DAGz8r2iV7M/7omI_zSA5tHtg4jRyYgMQA/edit) |
| 🔍 **KNN (K-Nearest Neighbors)** | [Xem Canvas](https://www.canva.com/design/DAGz-PdH2-U/1ZLbjFjGYQyIrQXhHAVDGg/edit) |
| 🌀 **KMeans Clustering** | [Xem Canvas](https://www.canva.com/design/DAG0AgMIdeI/nSC7YnyBhJq-lxUX4VHYwQ/edit) |
| 🧠 **CNN (Convolutional Neural Network)** | [Xem Canvas](https://www.canva.com/design/DAG0k1BtRMY/sSwAmH3TCnEO5oux_8C2iQ/edit) |
| 🗣️ **CNN for Text Classification (NLP)** | [Xem Canvas](https://www.canva.com/design/DAG0pMVsh1g/iLGh4JGtSY-XdLSzuF9Suw/edit) |
| 🧩 **VGG Network** | [Xem Canvas](https://www.canva.com/design/DAG1WmUVVAA/da9eJiuqoLMyywksKU9C_Q/edit) |
| 🎯 **YOLO (Object Detection)** | [Xem Canvas](https://www.canva.com/design/DAG13Fvc2Zo/YQOEeYHzBFIdm9_CjeqliA/edit) |
| 🔁 **ResNet (Residual Network)** | [Xem Canvas](https://www.canva.com/design/DAG2mWgdXkw/jjbjbrMHs3HSH7RTadEDMg/edit) |
| **Self-Organizing Maps (SOM) - Bản đồ tự tổ chức** | [Xem Canvas](https://www.canva.com/design/DAG3K5NC1QQ/N4HKT_5uaxu8NCR6T0w8Lg/edit?fbclid=IwY2xjawN4zxhleHRuA2FlbQIxMABicmlkETFiOVA5TjZxWG5IR0FaNkF5c3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHi4dnCu3wv2XU9drtbje9eYKgyC4VFfoDCfBk158SEl0khcs9wgdRm_wZBNu_aem_Z4unKlUHzVO50PFExkgYLA) |
| **RNN (Recurrent Neural Network)** | [Xem Canvas](https://www.canva.com/design/DAG2kjV7FP4/Nri7P3ejwkh9iYB9SnntOw/edit?fbclid=IwY2xjawN4z2hleHRuA2FlbQIxMABicmlkETFiOVA5TjZxWG5IR0FaNkF5c3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHgJlqoGyMQkOATqZOV0H941YliaTCdI03o381MtRCFTbsSM4UUMqmaocRd8x_aem_wkicuxCeA-39oCVUtMfCXA) |
| **EfficientNet** | [Xem Canvas](https://www.canva.com/design/DAG2kh8IXRI/cNZhpCeR4W0ft_sRqHGt1w/edit?ui=eyJEIjp7IlAiOnsiQiI6ZmFsc2V9fX0&fbclid=IwY2xjawN4z7JleHRuA2FlbQIxMABicmlkETFiOVA5TjZxWG5IR0FaNkF5c3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHpwGcX9lz8wJ-YS83lh0Cp0IC5Ct1X6dHsBrXRr0f0zixxlOrWhncCHV_y2k_aem_q4_TEyf3tW87hudHq5tEZg) |


---

> 🧱 **Ghi chú:** Danh sách đang được **cập nhật thêm...**  
> 💡 *Gợi ý:* Nên ghim trang này để tiện tra cứu khi học hoặc làm project.
