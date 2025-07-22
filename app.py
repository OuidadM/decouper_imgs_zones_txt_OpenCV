from flask import Flask, request, jsonify
from typing import List, Tuple
import cv2
import numpy as np
import base64
import openai
import re
import os

app = Flask(__name__)

# Configuration de l'API Key OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

def translate_image_with_gpt4o(base64_image: str, target_lang: str = "French"):
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

def detect_text_blocks(image_bytes, max_width=1200, max_height=1600, concat_direction='vertical'):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    h, w = img.shape[:2]
    if w > max_width or h > max_height:
        scaling_factor = min(max_width / w, max_height / h)
        new_size = (int(w * scaling_factor), int(h * scaling_factor))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Augmenter la lisibilite du texte
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    threshed = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 8
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    morphed = cv2.dilate(threshed, kernel, iterations=2)

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_blocks = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 60 and h > 20 and h < img.shape[0] * 0.6:
            pad = 6
            x1 = max(x - pad, 0)
            y1 = max(y - pad, 0)
            x2 = min(x + w + pad, img.shape[1])
            y2 = min(y + h + pad, img.shape[0])
            crop = img[y1:y2, x1:x2]
            candidate_blocks.append((x, y, crop))

    candidate_blocks.sort(key=lambda b: (b[1], b[0]))

    if not candidate_blocks:
        return [{"error": "Aucun bloc de texte détecté"}]

    cropped_images = []
    if concat_direction == 'vertical':
        target_width = max([b[2].shape[1] for b in candidate_blocks])
        for _, _, crop in candidate_blocks:
            h, w = crop.shape[:2]
            resized = cv2.copyMakeBorder(crop, 2, 2, 0, target_width - w, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            cropped_images.append(resized)
        merged_image = cv2.vconcat(cropped_images)
    else:
        target_height = max([b[2].shape[0] for b in candidate_blocks])
        for _, _, crop in candidate_blocks:
            h, w = crop.shape[:2]
            resized = cv2.copyMakeBorder(crop, 0, target_height - h, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            cropped_images.append(resized)
        merged_image = cv2.hconcat(cropped_images)

    _, buffer = cv2.imencode('.jpg', merged_image)
    encoded = base64.b64encode(buffer).decode()

    translated_text = translate_image_with_gpt4o(f"data:image/jpeg;base64,{encoded}", target_lang="French")

    return [{
        "merged_image": f"data:image/jpeg;base64,{encoded}",
        "translation": translated_text
    }]

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "Image file is missing"}), 400
    image = request.files["image"].read()
    blocks = detect_text_blocks(image, concat_direction='vertical')
    return jsonify({"blocks": blocks})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
