from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import openai
import os

app = Flask(__name__)

# Récupération de la clé OpenAI depuis une variable d'environnement
openai.api_key = os.getenv("OPENAI_API_KEY")


def translate_image_with_gpt4o(images, target_lang="French"):
    """
    Envoie les images au modèle GPT-4o pour une traduction directe.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un traducteur professionnel spécialisé dans la traduction officielle de documents administratifs et juridiques. "
                        f"Traduis fidèlement tous les éléments visibles de cette image en {target_lang}, sans reformulation ni omission. "
                        "Ne commente rien. Utilise un ton formel et juridique. "
                        "Si des parties sont illisibles, note-les comme [illisible]. N'inclus jamais le texte original. "
                        f"Réponds uniquement avec le texte traduit en {target_lang}."
                    )
                },
                {
                    "role": "user",
                    "content": images
                }
            ],
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Erreur GPT-4o : {str(e)}")


def detect_text_blocks(image_bytes, target_lang="French"):
    """
    Divise l'image en blocs horizontaux, encode en base64, et appelle la traduction.
    """
    try:
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Image invalide ou non décodable.")

        h, w = img.shape[:2]
        h_step = h // 4

        blocks = [img[i * h_step:(i + 1) * h_step, 0:w] for i in range(4)]
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

        translated_text = translate_image_with_gpt4o(images=image_prompts, target_lang=target_lang)

        return {"translation": translated_text}

    except Exception as e:
        raise RuntimeError(f"Erreur lors de la détection ou de l'encodage : {str(e)}")


@app.route("/")
def index():
    return "✅ API OK - Envoyez une image POST vers /detect"


@app.route("/detect", methods=["POST"])
def detect():
    """
    Endpoint principal pour recevoir une image, la découper, et retourner la traduction.
    """
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
    # PORT paramétrable pour compatibilité avec Render, etc.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
