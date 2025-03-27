import os
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import cv2

# Load mô hình đã huấn luyện
model_path = r"D:\WEB_AI\digit_recognition_optimized_colab.h5"
model = tf.keras.models.load_model(model_path)

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_from_directory(os.path.dirname(__file__), 'index.html')

def image_to_base64(img):
    """
    Chuyển đổi ảnh (numpy array) sang định dạng base64.
    """
    ret, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def preprocess_image(image):
    """
    Tiền xử lý ảnh và lưu lại các bước xử lý:
      1. Chuyển PIL image sang numpy array (grayscale)
      2. Inversion nếu nền quá sáng
      3. Ngưỡng (threshold) tạo ảnh nhị phân
      4. Dilation để làm dày nét chữ mảnh
      5. Tìm contour, xác định bounding box và crop
      6. Resize để chữ số chiếm tối đa 20x20
      7. Padding về kích thước 28x28
      8. Chuẩn hóa ảnh về khoảng [0,1]
      
    Hàm trả về:
      - final_image: ảnh đã được chuẩn hóa, shape (1,28,28,1) để đưa vào mô hình
      - steps: dictionary chứa các bước xử lý dưới dạng ảnh (numpy array)
    """
    steps = {}
    
    # Bước 1: Ảnh gốc
    img = np.array(image)
    steps["original"] = img.copy()
    
    # Bước 2: Inversion nếu cần (nếu nền sáng)
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)
    steps["inverted"] = img.copy()
    
    # Bước 3: Ngưỡng (threshold) để tạo ảnh nhị phân
    _, img_thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    steps["threshold"] = img_thresh.copy()
    
    # Bước 4: Dilation để làm dày nét chữ mảnh
    kernel = np.ones((3, 3), np.uint8)  # Tăng kích thước kernel từ (2, 2) lên (3, 3)
    img_dilated = cv2.dilate(img_thresh, kernel, iterations=2)  # Tăng số lần lặp từ 1 lên 2
    steps["dilated"] = img_dilated.copy()
    
    # Bước 5: Tìm contour và xác định bounding box
    contours, _ = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Chọn contour có diện tích lớn nhất
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        # Vẽ bounding box trên ảnh đã inversion (cho mục đích hiển thị)
        img_box = img.copy()
        cv2.rectangle(img_box, (x, y), (x+w, y+h), (128, 128, 128), 2)
        steps["bounding_box"] = img_box.copy()
        # Crop vùng chứa chữ số
        img_cropped = img[y:y+h, x:x+w]
        steps["cropped"] = img_cropped.copy()
    else:
        img_cropped = img.copy()
        steps["cropped"] = img_cropped.copy()
    
    # Bước 6: Resize ảnh sao cho chữ số có kích thước tối đa 20x20 (giữ tỉ lệ)
    h, w = img_cropped.shape
    scale = 20 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img_cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    steps["resized"] = img_resized.copy()
    
    # Bước 7: Padding ảnh về kích thước 28x28 để căn giữa chữ số
    padded_img = np.zeros((28, 28), dtype=np.uint8)
    pad_w = (28 - new_w) // 2
    pad_h = (28 - new_h) // 2
    padded_img[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = img_resized
    steps["final"] = padded_img.copy()
    
    # Bước 8: Chuẩn hóa ảnh về [0,1] và điều chỉnh shape cho model
    final_image = padded_img.astype("float32") / 255.0
    final_image = np.expand_dims(final_image, axis=-1)  # (28,28,1)
    final_image = np.expand_dims(final_image, axis=0)     # (1,28,28,1)
    
    return final_image, steps

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Lấy dữ liệu ảnh từ base64 (loại bỏ header)
        image_data = data['image'].split(",")[1]
        img = Image.open(BytesIO(base64.b64decode(image_data))).convert('L')
        
        # Lưu ảnh gốc để hiển thị
        buffered_original = BytesIO()
        img.save(buffered_original, format="PNG")
        img_original_base64 = base64.b64encode(buffered_original.getvalue()).decode('utf-8')
        
        # Tiền xử lý ảnh và lưu lại các bước
        img_array, steps = preprocess_image(img)
        
        # Chuyển các bước xử lý sang định dạng base64 để gửi trả về
        steps_base64 = {}
        for key, step_img in steps.items():
            # Nếu ảnh có 1 channel thì chuyển về định dạng màu xám
            if len(step_img.shape) == 2:
                steps_base64[key] = "data:image/png;base64," + image_to_base64(step_img)
            else:
                steps_base64[key] = "data:image/png;base64," + image_to_base64(step_img)
        
        # Dự đoán với mô hình
        prediction = model.predict(img_array)
        predicted_label = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)
        
        # Log thông tin dự đoán
        app.logger.info(f"Predicted: {predicted_label}, Confidence: {confidence:.2f}%")
        
        return jsonify({
            'digit': predicted_label,
            'confidence': confidence,
            'original_image': "data:image/png;base64," + img_original_base64,
            'steps': steps_base64
        })
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
