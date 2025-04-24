import os
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
# Import các hàm tiền xử lý từ src.data.preprocessing
from src.data.preprocessing import (
    image_to_base64,
    apply_gaussian_blur,
    enhance_black_background,
    resize_and_pad,
    group_boxes,
    increase_stroke_thickness,
    preprocess_image
)

# Đường dẫn đến mô hình đã huấn luyện
model_path = "saved_models/digit_recognition_optimized_colab.h5"
model = tf.keras.models.load_model(model_path)

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_from_directory(os.path.dirname(__file__), 'index.html')

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

if __name__ == '__main__':
    app.run(debug=True)
