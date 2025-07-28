from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import openai
import os

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")


def translate_image_with_gpt4o(text, target_lang="French"):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un traducteur professionnel spécialisé dans la traduction officielle de documents administratifs et juridiques. "
                    "Traduis fidèlement et exactement tous les éléments du texte original en respectant le ton formel et administratif. "
                    f"Répond uniquement en {target_lang}."
                    "Ne simplifie pas, ne reformule pas, n'interprète rien, ne commente rien. "
                    "Conserve la structure logique, les noms propres, les dates, les références de décrets, et les termes juridiques ou institutionnels. "
                    "Évite toute approximation. Si une date ou un nom n'est pas lisible, indique [illisible] sans essayer de le deviner. "
                    f"La traduction doit être entièrement en {target_lang}. "
                    "N'inclus jamais le texte original en langue originale. Ne laisse aucun passage non traduit."
                )
            },
            {
                "role": "user",
                "content": text
            }
        ],
        max_tokens=2000
    )
    return response.choices[0].message.content.strip()

def detect_text_blocks(image_bytes, target_lang="French"):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    h, w = img.shape[:2]
    h_step = h // 4

    # Découpage en 4 blocs horizontaux
    blocks = [img[i*h_step:(i+1)*h_step, 0:w] for i in range(4)]
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

        # Décommenter si tu veux enregistrer les blocs localement pour debug
        # cv2.imwrite(f"block_{idx+1}.png", block)

    translated_text = translate_image_with_gpt4o(images=image_prompts, target_lang=target_lang)

    return {
        "translation": translated_text
    }


@app.route("/")
def index():
    return "API OK - POST une image vers /detect pour la traduction."


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "Image file is missing"}), 400

    image_bytes = request.files["image"].read()
    target_lang = request.args.get("lang", "French")

    try:
        result = detect_text_blocks(image_bytes, target_lang)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
