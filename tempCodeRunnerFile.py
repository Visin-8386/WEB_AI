import os
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import tempfile
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import cv2

# Load mô hình đã huấn luyện
model_path = r"D:\WEB_AI\digit_recognition_optimized.h5"
model = tf.keras.models.load_model(model_path)

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_from_directory(os.path.dirname(__file__), 'index.html')

def image_to_base64(img: np.ndarray) -> str:
    """
    Chuyển đổi ảnh (numpy array) sang định dạng base64.
    """
    ret, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def preprocess_image(image_path: str):
    """
    Xử lý ảnh từ file với các bước:
      1. Đọc ảnh dưới dạng grayscale.
      2. Kiểm tra nền ảnh: nếu nền sáng (mean > 127) => đảo màu, còn không thì giữ nguyên.
      3. Chuẩn hóa ảnh bằng cv2.normalize.
      4. Áp dụng threshold (đã nâng từ 30 lên 50) tạo ảnh nhị phân.
      5. Dilation (làm dày nét) để giữ nét chữ rõ hơn.
      6. Tìm bounding box chứa chữ số và crop.
      7. Padding ảnh để tạo ảnh vuông.
      8. Resize ảnh về kích thước 28x28.
      9. Chuẩn hóa ảnh về [0,1] và điều chỉnh shape cho model.
      
    Trả về:
      - final_img: ảnh cuối cùng dạng (1,28,28,1) (dùng cho mô hình)
      - steps: dictionary chứa các bước xử lý (dạng numpy array) để gửi về client
    """
    steps = {}

    # Bước 1: Đọc ảnh grayscale và lưu ảnh gốc
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    steps["original"] = img.copy()

    # Bước 2: Kiểm tra nền ảnh và đảo màu nếu cần
    if np.mean(img) > 127:
        img_inverted = cv2.bitwise_not(img)
        steps["inversion"] = img_inverted.copy()
    else:
        img_inverted = img.copy()
        steps["inversion"] = img_inverted.copy()

    # Bước 3: Chuẩn hóa ảnh
    img_normalized = cv2.normalize(img_inverted, None, 0, 255, cv2.NORM_MINMAX)
    steps["normalized"] = img_normalized.copy()

    # Bước 4: Áp dụng threshold => chữ trắng nổi bật hơn
    # (tăng từ 30 lên 50)
    _, thresh = cv2.threshold(img_normalized, 50, 255, cv2.THRESH_BINARY)
    steps["threshold"] = thresh.copy()

    # Bước 5: Dilation - làm dày nét chữ
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    steps["dilated"] = dilated.copy()

    # Bước 6: Tìm bounding box và crop chữ số (trên ảnh dilated)
    x, y, w, h = cv2.boundingRect(dilated)
    cropped = img_normalized[y:y+h, x:x+w]  # vẫn cắt trên img_normalized để giữ cường độ
    steps["cropped"] = cropped.copy()

    # Bước 7: Padding để tạo ảnh vuông
    size = max(w, h)
    padded = np.ones((size, size), dtype=np.uint8) * 0
    dx, dy = (size - w) // 2, (size - h) // 2
    padded[dy:dy+h, dx:dx+w] = cropped
    steps["padded"] = padded.copy()

    # Bước 8: Resize ảnh về 28x28
    resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)
    steps["resized"] = resized.copy()

    # Bước 9: Chuẩn hóa ảnh về [0,1] và điều chỉnh shape cho model
    final_img = resized.astype("float32") / 255.0
    final_img = np.expand_dims(final_img, axis=-1)  # (28,28,1)
    final_img = np.expand_dims(final_img, axis=0)   # (1,28,28,1)

    # Lưu ảnh 28x28 cuối trước khi chuẩn hóa (dùng để tạo ảnh phóng to)
    steps["final"] = resized.copy()

    return final_img, steps

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Lấy dữ liệu ảnh từ base64 (loại bỏ header)
        image_data = data['image'].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert('L')

        # Lưu ảnh gốc vào file tạm thời để sử dụng preprocess_image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_filename = tmp.name
            image.save(temp_filename, format="PNG")

        # Gọi hàm xử lý ảnh và lấy các bước xử lý
        img_array, steps = preprocess_image(temp_filename)

        # Xóa file tạm sau khi xử lý
        os.remove(temp_filename)

        # Dự đoán với mô hình
        preds = model.predict(img_array)[0]  # shape (10,)
        predicted_label = int(np.argmax(preds))
        confidence = float(np.max(preds) * 100)

        # ----- Lấy top-3 dự đoán để xem mô hình "phân vân" ra sao -----
        sorted_indices = np.argsort(preds)[::-1]  # sắp xếp giảm dần
        top3_indices = sorted_indices[:3]
        top3_values = preds[top3_indices]  # Xác suất (0..1)
        top3_info = [
            {
                "digit": int(d),
                "confidence": float(c * 100)
            }
            for d, c in zip(top3_indices, top3_values)
        ]

        # ----- Tạo ảnh phóng to để vẽ kết quả -----
        # Ảnh "final" là ảnh 28x28 gốc (uint8), ta phóng to lên để vẽ text
        display_size = 112  # phóng to gấp 4 lần (28 * 4)
        result_img_big = cv2.resize(
            steps["final"], (display_size, display_size),
            interpolation=cv2.INTER_NEAREST
        )
        text = f"{predicted_label} ({confidence:.2f}%)"
        cv2.putText(
            result_img_big,
            text,
            (5, display_size - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Chuyển đổi các bước xử lý sang định dạng base64 để gửi về
        steps_base64 = {}
        for key, step_img in steps.items():
            steps_base64[key] = "data:image/png;base64," + image_to_base64(step_img)

        # Ảnh "predicted" là ảnh phóng to có vẽ nhãn
        steps_base64["predicted"] = "data:image/png;base64," + image_to_base64(result_img_big)

        # Chuyển ảnh gốc sang base64 để trả về
        buffered_original = BytesIO()
        image.save(buffered_original, format="PNG")
        img_original_base64 = base64.b64encode(buffered_original.getvalue()).decode('utf-8')

        app.logger.info(f"Predicted: {predicted_label}, Confidence: {confidence:.2f}%")

        return jsonify({
            'digit': predicted_label,
            'confidence': confidence,
            'top3': top3_info,  # Gửi luôn top-3 để client xem
            'original_image': "data:image/png;base64," + img_original_base64,
            'steps': steps_base64
        })
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
