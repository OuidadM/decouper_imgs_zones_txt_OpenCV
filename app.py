from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import openai
import re
import os

app = Flask(__name__)

# Configuration de l'API Key OpenAI (à configurer dans les variables d’environnement de Render)
openai.api_key = os.getenv("OPENAI_API_KEY")


def translate_image_with_gpt4o(base64_image: str, target_lang: str = "French"):
    """
    Envoie une image encodée en base64 à GPT-4o pour traduction directe.
    """
    base64_clean = re.sub("^data:image/.+;base64,", "", base64_image)

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"Tu es un traducteur expert. Traduis uniquement le texte de l'image fournie en {target_lang}, sans ajouter de commentaires."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_clean}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Erreur OpenAI] {str(e)}"

def detect_text_blocks(image_bytes):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Binarisation adaptative
    threshed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 15, 10)

    # 2. Morphologie – kernel plus fin pour du texte arabe dense
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 4))
    morphed = cv2.dilate(threshed, kernel, iterations=1)

    # 3. Détection des contours
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blocks = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # 4. Filtrage plus souple des zones textuelles
        aspect_ratio = w / h
        if 60 < w < img.shape[1] and 20 < h < img.shape[0] // 2 and 1.0 < aspect_ratio < 15.0:
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

    # 5. Tri des blocs par ordre de lecture (du haut vers le bas)
    blocks = sorted(blocks, key=lambda b: b["y"])

    return blocks


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "Image file is missing"}), 400
    image = request.files["image"].read()
    blocks = detect_text_blocks(image)
    return jsonify({"blocks": blocks})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
