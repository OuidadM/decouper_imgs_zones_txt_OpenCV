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
    """
    Détecte les blocs de texte dans une image et les retourne avec leur traduction.
    """
    # Chargement et prétraitement de l’image
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Binarisation adaptative pour supporter différents niveaux d’éclairage
    threshed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 10
    )

    # 2. Dilatation pour regrouper les mots sur une ligne
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5))
    morphed = cv2.dilate(threshed, kernel, iterations=1)

    # 3. Détection des contours externes
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blocks = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # 4. Filtrer les blocs non pertinents
        if w > 80 and h > 20 and h < img.shape[0] * 0.5 and w < img.shape[1] * 0.95:
            crop = img[y:y+h, x:x+w]
            _, buffer = cv2.imencode('.jpg', crop)
            encoded = base64.b64encode(buffer).decode()

            # Traduction avec GPT-4o
            translated_text = translate_image_with_gpt4o(f"data:image/jpeg;base64,{encoded}", target_lang="French")

            blocks.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "image": f"data:image/jpeg;base64,{encoded}",
                "translation": translated_text
            })

    # 5. Tri des blocs de haut en bas puis de gauche à droite
    blocks.sort(key=lambda b: (b["y"], b["x"]))
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
