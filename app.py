from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

def detect_text_blocks(image_bytes):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Binarisation adaptative pour gérer différentes zones d'éclairage
    threshed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 15, 10)

    # 2. Morphologie pour connecter les zones de texte
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    morphed = cv2.dilate(threshed, kernel, iterations=1)

    # 3. Détection des contours
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blocks = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 4. Filtrage des zones non textuelles : dimensions minimales/maximales
        if 60 < w < img.shape[1] and 20 < h < img.shape[0] // 2:
            # Ignorer les très grands blocs (ex. logos centrés ou en haut)
            crop = img[y:y+h, x:x+w]
            _, buffer = cv2.imencode('.jpg', crop)
            encoded = base64.b64encode(buffer).decode()
            blocks.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "image": f"data:image/jpeg;base64,{encoded}"
            })

    return blocks


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "Image file is missing"}), 400
    image = request.files["image"].read()
    blocks = detect_text_blocks(image)
    return jsonify({"blocks": blocks})


