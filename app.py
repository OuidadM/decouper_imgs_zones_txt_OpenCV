from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import openai
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
                    f"Tu es un traducteur professionnel. Traduis tout son contenu fidèlement en {target_lang}, même s'il s'agit d'un document officiel. "
                    f"N'inclus pas de texte de langue originale dans ta réponse, uniquement la version traduite en {target_lang}. Ignore les doublons visuels ou décoratifs. "
                    "L'image ne contient ni personnes, ni visages, ni informations personnelles. "
                    "Ignore les éléments graphiques. Ne commente rien. Fournis uniquement la traduction du texte présent."
                )
            },
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
