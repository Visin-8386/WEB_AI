import os
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Đường dẫn đến mô hình đã huấn luyện
model_path = "digit_recognition_optimized_colab.h5"
model = tf.keras.models.load_model(model_path)

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_from_directory(os.path.dirname(__file__), 'index.html')

def image_to_base64(img: np.ndarray) -> str:
    """Chuyển ảnh numpy array sang chuỗi base64 để hiển thị trên web."""
    ret, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    """Áp dụng Gaussian blur để giảm nhiễu."""
    return cv2.GaussianBlur(image, kernel_size, sigma)

def enhance_black_background(img_gray):
    """
    Xử lý nền để đảm bảo nền đen và chữ số trắng:
      1. Áp dụng Otsu threshold trực tiếp trên ảnh đã làm mờ.
      2. Đảo ngược màu nếu cần (để chữ số thành trắng, nền đen).
      3. Làm sạch kết quả bằng phép toán morphology.
    """
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    threshold, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pixels_below_threshold = np.sum(hist[:int(threshold)])
    pixels_above_threshold = np.sum(hist[int(threshold):])
    if pixels_above_threshold > pixels_below_threshold:
        binary = cv2.bitwise_not(binary)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return binary

def resize_and_pad(roi, size=28, padding=5):
    """
    Resize ROI theo tỉ lệ ban đầu để chữ số không bị biến dạng,
    sau đó chèn vào canvas đen kích thước size x size, căn giữa chữ số.
    """
    h, w = roi.shape
    if h > w:
        new_h = size - 2 * padding
        new_w = int(w * (new_h / h))
    else:
        new_w = size - 2 * padding
        new_h = int(h * (new_w / w))
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def group_boxes(digit_boxes, row_threshold=20):
    """
    Gom các bounding box vào từng dòng dựa trên giá trị trung tâm theo chiều dọc.
    """
    digit_boxes.sort(key=lambda b: b[1])
    rows = []
    for box in digit_boxes:
        x, y, w, h = box
        center_y = y + h / 2
        if not rows:
            rows.append([box])
        else:
            last_row = rows[-1]
            avg_center_y = np.mean([r[1] + r[3] / 2 for r in last_row])
            if abs(center_y - avg_center_y) <= row_threshold:
                last_row.append(box)
            else:
                rows.append([box])
    for r in rows:
        r.sort(key=lambda b: b[0])
    sorted_boxes = []
    for r in rows:
        sorted_boxes.extend(r)
    return sorted_boxes

def increase_stroke_thickness(roi, dilation_kernel_size=(5, 5), dilation_iterations=3, padding=10):
    """
    Tăng độ dày nét vẽ và làm bự chữ số bằng cách áp dụng dilation luôn.
    
    Args:
        roi: Ảnh ROI chứa chữ số.
        dilation_kernel_size: Kích thước kernel cho dilation (ví dụ (5,5)).
        dilation_iterations: Số lần áp dụng dilation.
        padding: Số pixel padding thêm vào xung quanh chữ số.
    
    Returns:
        ROI sau khi đã tăng độ dày nét và thêm padding.
    """
    dilation_kernel = np.ones(dilation_kernel_size, np.uint8)
    roi_dilated = cv2.dilate(roi, dilation_kernel, iterations=dilation_iterations)
    roi_padded = cv2.copyMakeBorder(roi_dilated, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    return roi_padded

def preprocess_image(image):
    """
    Tiền xử lý ảnh vẽ (nền đen, số trắng) theo các bước:
      1) Chuyển ảnh sang grayscale.
      2) Làm mờ ảnh để giảm nhiễu.
      3) Xử lý nền để đảm bảo nền đen, số trắng.
      4) Tìm contours ngoài để lấy vùng chứa số.
      5) Mở rộng bounding box để không cắt sát mép.
      6) Gom các bounding box thành các dòng.
      7) Với mỗi bounding box:
         - Nếu nét quá nhỏ so với kích thước trung bình, tăng độ dày nét (dilation) và thêm padding.
         - Resize về 28x28 và chuẩn hóa [0,1].
      8) Trả về danh sách ảnh đã xử lý và dictionary chứa các bước xử lý.
    """
    steps = {}

    # 1. Chuyển ảnh sang grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    steps["gray"] = image_to_base64(img_gray)

    # 2. Làm mờ ảnh để giảm nhiễu
    img_blurred = apply_gaussian_blur(img_gray, kernel_size=(5, 5))
    steps["blurred"] = image_to_base64(img_blurred)

    # 3. Xử lý nền để đảm bảo nền đen, số trắng
    img_optimized = enhance_black_background(img_blurred)
    steps["optimized_background"] = image_to_base64(img_optimized)

    # 4. Tìm contours ngoài để lấy vùng chứa số
    contours, _ = cv2.findContours(img_optimized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_contours = cv2.cvtColor(img_optimized.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)
    steps["contours"] = image_to_base64(img_with_contours)

    # 5. Mở rộng bounding box để không cắt sát mép
    digit_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 5 and cv2.contourArea(cnt) > 25:
            pad = 5
            x_pad = max(0, x - pad)
            y_pad = max(0, y - pad)
            w_pad = w + 2 * pad
            h_pad = h + 2 * pad
            if x_pad + w_pad > img_optimized.shape[1]:
                w_pad = img_optimized.shape[1] - x_pad
            if y_pad + h_pad > img_optimized.shape[0]:
                h_pad = img_optimized.shape[0] - y_pad
            digit_boxes.append((x_pad, y_pad, w_pad, h_pad))

    # 6. Gom các bounding box thành các dòng và sắp xếp
    img_with_boxes = cv2.cvtColor(img_optimized.copy(), cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in digit_boxes:
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)
    steps["boxes"] = image_to_base64(img_with_boxes)

    digit_boxes = group_boxes(digit_boxes, row_threshold=20)
    img_with_sorted_boxes = cv2.cvtColor(img_optimized.copy(), cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(digit_boxes):
        cv2.rectangle(img_with_sorted_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img_with_sorted_boxes, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    steps["sorted_boxes"] = image_to_base64(img_with_sorted_boxes)

    # 7. Xử lý từng bounding box: tăng nét nếu nét quá nhỏ so với trung bình
    processed_digits = []
    steps["digits_raw"] = []
    steps["thickened_digits"] = []
    steps["digits_resized"] = []
    steps["is_thickened_digits"] = []

    # Lấy danh sách ROI từ các bounding box
    roi_list = []
    for (x, y, w, h) in digit_boxes:
        roi = img_optimized[y:y+h, x:x+w]
        roi_list.append(roi)

    # Ngưỡng cố định: nếu chiều cao lớn hơn 350 pixel thì tăng nét
    HEIGHT_THRESHOLD = 350

    for idx, roi in enumerate(roi_list):
        steps["digits_raw"].append(image_to_base64(roi))
        h = roi.shape[0]
        # Tăng nét nếu chiều cao ROI > HEIGHT_THRESHOLD
        if h > HEIGHT_THRESHOLD:
            roi_thickened = increase_stroke_thickness(
                roi, dilation_kernel_size=(5, 5), dilation_iterations=2, padding=6)
            steps["thickened_digits"].append(image_to_base64(roi_thickened))
            steps["is_thickened_digits"].append(True)
        else:
            roi_thickened = roi
            steps["is_thickened_digits"].append(False)

        roi_resized = resize_and_pad(roi_thickened, size=28, padding=5)
        steps["digits_resized"].append(image_to_base64(roi_resized))

        roi_norm = roi_resized.astype("float32") / 255.0
        roi_norm = np.expand_dims(roi_norm, axis=-1)  # (28,28) -> (28,28,1)
        roi_norm = np.expand_dims(roi_norm, axis=0)   # (28,28,1) -> (1,28,28,1)
        processed_digits.append(roi_norm)

    return processed_digits, steps

@app.route('/predict_multiple', methods=['POST'])
def predict_multiple():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'Không có ảnh được gửi!'}), 400
        
        image_data = data['image'].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        
        digit_images, steps = preprocess_image(image_np)
        if not digit_images:
            return jsonify({'error': 'Không tìm thấy số nào trong ảnh!'}), 400
        
        predicted_digits = []
        confidence_scores = []
        for digit_img in digit_images:
            preds = model.predict(digit_img, verbose=0)[0]
            predicted_digit = int(np.argmax(preds))
            confidence = float(preds[predicted_digit])
            predicted_digits.append(predicted_digit)
            confidence_scores.append(confidence)
        
        number_string = "".join(map(str, predicted_digits))
        return jsonify({
            'digits': predicted_digits,
            'confidence': confidence_scores,
            'number_string': number_string,
            'steps': steps
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test_image_processing', methods=['POST'])
def test_image_processing():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'Không có ảnh được gửi!'}), 400
        
        image_data = data['image'].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        blurred = apply_gaussian_blur(gray)
        enhanced = enhance_black_background(blurred)
        
        return jsonify({
            'original_gray': image_to_base64(gray),
            'gaussian_blur': image_to_base64(blurred),
            'enhanced_background': image_to_base64(enhanced)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
