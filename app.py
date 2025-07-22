from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

def detect_text_blocks(image_bytes):
    # Décodage image
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Binarisation adaptative pour supporter différents niveaux d’éclairage
    threshed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 10
    )

    # 2. Dilatation pour regrouper les lettres sur la même ligne
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5))  # Largeur augmentée
    morphed = cv2.dilate(threshed, kernel, iterations=1)

    # 3. Détection des contours externes
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blocks = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # 4. Filtrer les zones non pertinentes
        if w > 80 and h > 20 and h < img.shape[0] * 0.5 and w < img.shape[1] * 0.95:
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

    # 5. Tri des blocs de haut en bas
    blocks.sort(key=lambda b: (b["y"], b["x"]))
    return blocks


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "Image file is missing"}), 400
    image = request.files["image"].read()
    blocks = detect_text_blocks(image)
    return jsonify({"blocks": blocks})
