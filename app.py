import os
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Đường dẫn đến mô hình đã huấn luyện (đảm bảo mô hình của bạn được huấn luyện với dạng số trắng trên nền đen)
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
    
    # Tạo canvas đen và chèn ảnh đã resize vào giữa
    canvas = np.zeros((size, size), dtype=np.uint8)
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def group_boxes(digit_boxes, row_threshold=20):
    """
    Gom các bounding box vào từng dòng dựa trên giá trị trung tâm theo chiều dọc (center y).
    Nếu độ chênh lệch giữa center_y của các box nhỏ hơn row_threshold thì coi là cùng dòng.
    Sau đó, sắp xếp từng dòng theo tọa độ x (trái qua phải) và ghép lại theo thứ tự dòng (trên xuống dưới).
    """
    # Sắp xếp sơ bộ theo y
    digit_boxes.sort(key=lambda b: b[1])  # b[1] là y
    rows = []  # Mỗi phần tử là list chứa các box thuộc cùng một dòng

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
    # Sắp xếp từng dòng theo x
    for r in rows:
        r.sort(key=lambda b: b[0])
    # Ghép các dòng theo thứ tự từ trên xuống dưới
    sorted_boxes = []
    for r in rows:
        sorted_boxes.extend(r)
    return sorted_boxes

def preprocess_image(image):
    """
    Tiền xử lý ảnh vẽ (với nền đen, số màu trắng) theo các bước:
      1) Chuyển ảnh sang grayscale.
      2) Áp dụng Otsu threshold.
      3) Tìm contours ngoài để lấy vùng chứa số.
      4) Với mỗi contour đủ lớn, mở rộng bounding box (padding) để không cắt sát mép.
      5) Sử dụng hàm group_boxes để gom các bounding box thành từng dòng:
         - Nếu tọa độ y không chênh lệch quá nhiều, sắp xếp theo thứ tự trái qua phải.
      6) Resize mỗi ROI về 28x28 (không bóp méo) và chuẩn hóa [0,1].
      7) Trả về danh sách ảnh đã xử lý và dictionary steps chứa các bước xử lý (base64).
    """
    steps = {}

    # 1. Chuyển ảnh sang grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    steps["gray"] = image_to_base64(img_gray)

    # 2. Otsu threshold
    _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    steps["threshold"] = image_to_base64(img_thresh)

    # 3. Tìm contours ngoài
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Tạo bản sao của ảnh để vẽ contours lên
    img_with_contours = cv2.cvtColor(img_thresh.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)
    steps["contours"] = image_to_base64(img_with_contours)

    digit_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Lọc bỏ nhiễu: chỉ giữ contour có kích thước đủ lớn
        if w > 5 and h > 5:
            pad = 5
            x_pad = max(0, x - pad)
            y_pad = max(0, y - pad)
            w_pad = w + 2 * pad
            h_pad = h + 2 * pad
            # Đảm bảo không vượt quá biên ảnh
            if x_pad + w_pad > img_thresh.shape[1]:
                w_pad = img_thresh.shape[1] - x_pad
            if y_pad + h_pad > img_thresh.shape[0]:
                h_pad = img_thresh.shape[0] - y_pad
            digit_boxes.append((x_pad, y_pad, w_pad, h_pad))

    # Tạo ảnh với bounding boxes
    img_with_boxes = cv2.cvtColor(img_thresh.copy(), cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in digit_boxes:
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)
    steps["boxes"] = image_to_base64(img_with_boxes)

    # 5. Gom và sắp xếp bounding box theo từng dòng
    digit_boxes = group_boxes(digit_boxes, row_threshold=20)

    # Tạo ảnh với bounding boxes đã sắp xếp
    img_with_sorted_boxes = cv2.cvtColor(img_thresh.copy(), cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(digit_boxes):
        cv2.rectangle(img_with_sorted_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Thêm số thứ tự để thấy rõ thứ tự sắp xếp
        cv2.putText(img_with_sorted_boxes, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    steps["sorted_boxes"] = image_to_base64(img_with_sorted_boxes)

    raw_digits = []
    processed_digits = []
    steps["digits_raw"] = []
    steps["digits_resized"] = []

    for (x, y, w, h) in digit_boxes:
        # Cắt ROI
        roi = img_thresh[y:y+h, x:x+w]
        steps["digits_raw"].append(image_to_base64(roi))
        # 6. Resize ROI về 28x28 mà không bóp méo
        roi_resized = resize_and_pad(roi, size=28, padding=5)
        steps["digits_resized"].append(image_to_base64(roi_resized))
        # Chuẩn hóa về [0,1] và định dạng lại cho mô hình
        roi_norm = roi_resized.astype("float32") / 255.0
        roi_norm = np.expand_dims(roi_norm, axis=-1)  # (28,28,1)
        roi_norm = np.expand_dims(roi_norm, axis=0)    # (1,28,28,1)
        processed_digits.append(roi_norm)

    return processed_digits, steps

@app.route('/predict_multiple', methods=['POST'])
def predict_multiple():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'Không có ảnh được gửi!'}), 400

        # Lấy chuỗi base64 từ dataURL (loại bỏ phần "data:image/png;base64,")
        image_data = data['image'].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)

        # Tiền xử lý ảnh
        digit_images, steps = preprocess_image(image_np)
        if not digit_images:
            return jsonify({'error': 'Không tìm thấy số nào trong ảnh!'}), 400

        # Dự đoán từng chữ số
        predicted_digits = []
        for digit_img in digit_images:
            preds = model.predict(digit_img)[0]
            predicted_digit = int(np.argmax(preds))
            predicted_digits.append(predicted_digit)

        number_string = "".join(map(str, predicted_digits))
        return jsonify({
            'digits': predicted_digits,
            'number_string': number_string,
            'steps': steps
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
