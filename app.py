from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import openai
import re
import os

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

def translate_image_with_gpt4o(images, target_lang="French"):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un traducteur professionnel spécialisé dans la traduction officielle de documents administratifs et juridiques. "
                    "L'image contient du texte en arabe, probablement issu d'un document officiel du gouvernement marocain. "
                    "Traduis fidèlement et exactement tous les éléments du texte original, en respectant le ton formel et administratif. "
                    "Ne simplifie pas, ne reformule pas. Ne commente rien. "
                    "Conserve la structure logique, les noms propres, les dates, les références de décrets, et les termes juridiques ou institutionnels tels quels. "
                    "Évite toute approximation. Si une date ou un nom n'est pas lisible, indique [illisible] sans essayer de le deviner. "
                    f"La traduction doit être entièrement en {target_lang}. N'inclus pas de texte arabe. Ne reformule pas en langage courant."
                )
            }
            ,
            {
                "role": "user",
                "content": images
            }
        ],
        max_tokens=1000,
    )

    return response.choices[0].message.content.strip()

def detect_text_blocks(image_bytes, target_lang="French"):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    h, w = img.shape[:2]

    # Diviser en 4 bandes horizontales égales
    h_step = h // 4

    blocks = [
        img[i*h_step:(i+1)*h_step, 0:w] for i in range(4)
    ]

    image_prompts = []

    for idx, block in enumerate(blocks):
        _, buffer = cv2.imencode('.png', block)
        encoded = base64.b64encode(buffer).decode("utf-8")
        image_prompts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encoded}"
            }
        })

    # Envoi à GPT-4o avec tous les blocs dans un seul message
    translated_text = translate_image_with_gpt4o(images=image_prompts, target_lang=target_lang)
    return {
        "translation": translated_text
    }


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "Image file is missing"}), 400
    image = request.files["image"].read()
    blocks=detect_text_blocks(image)
    return jsonify(blocks)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))