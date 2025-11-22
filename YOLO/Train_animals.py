import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- 1. CẤU HÌNH CÁC THAM SỐ ---
# Cập nhật đường dẫn này tới thư mục chứa bộ dữ liệu đã giải nén
data_dir = 'datasets/animals/animals' # <-- !!! THAY ĐỔI ĐƯỜNG DẪN NÀY !!!datasets\animals\animals

# Các tham số cho hình ảnh và huấn luyện
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32  # Batch size vừa đủ cho hầu hết các GPU
EPOCHS = 50
NUM_CLASSES = 90 # Số lượng loài động vật trong dataset

# --- 2. CHUẨN BỊ DỮ LIỆU ---
# Tạo các trình tạo dữ liệu (Data Generator) để tải và tăng cường dữ liệu
# Data augmentation giúp mô hình khái quát hóa tốt hơn
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Dành 20% dữ liệu cho việc kiểm định (validation)
)

# Trình tạo dữ liệu cho tập huấn luyện
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training' # Chỉ định đây là tập huấn luyện
)

# Trình tạo dữ liệu cho tập kiểm định
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation' # Chỉ định đây là tập kiểm định
)

# --- 3. XÂY DỰNG MÔ HÌNH (HỌC CHUYỂN GIAO) ---
# Tải mô hình MobileNetV2 đã được huấn luyện trước, không bao gồm lớp phân loại trên cùng
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Đóng băng các lớp của mô hình gốc để không huấn luyện lại chúng trong giai đoạn đầu
base_model.trainable = False

# Thêm các lớp phân loại của riêng bạn vào trên cùng
x = base_model.output
x = GlobalAveragePooling2D()(x) # Lớp gộp trung bình toàn cục
x = Dense(1024, activation='relu')(x) # Một lớp kết nối đầy đủ
predictions = Dense(NUM_CLASSES, activation='softmax')(x) # Lớp output với 90 units và softmax

# Tạo mô hình hoàn chỉnh
model = Model(inputs=base_model.input, outputs=predictions)

# --- 4. BIÊN DỊCH (COMPILE) MÔ HÌNH ---
# Sử dụng Adam optimizer và categorical_crossentropy cho bài toán phân loại đa lớp
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# In cấu trúc của mô hình
model.summary()


# --- 5. HUẤN LUYỆN MÔ HÌNH ---
print("\nBắt đầu quá trình huấn luyện...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=validation_generator.samples // BATCH_SIZE
)
print("Hoàn tất huấn luyện!")

# --- 6. ĐÁNH GIÁ VÀ TRỰC QUAN HÓA KẾT QUẢ ---
# Lấy các giá trị accuracy và loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

# Vẽ biểu đồ Accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Vẽ biểu đồ Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# --- 7. LƯU MÔ HÌNH (TÙY CHỌN) ---
model.save('animal_classifier_model.h5')
print("Mô hình đã được lưu với tên animal_classifier_model.h5")