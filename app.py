from flask import Flask, request, jsonify
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from markdown2 import markdown as md2html
import requests
import os
import io
import time
import base64
import re

app = Flask(__name__)

# Variables d'environnement
AZURE_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_VISION_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Client Azure Computer Vision
cv_client = ComputerVisionClient(AZURE_ENDPOINT, CognitiveServicesCredentials(AZURE_KEY))

# OCR Azure
def extract_text_azure(image_bytes):
    image_stream = io.BytesIO(image_bytes)
    response = cv_client.read_in_stream(image=image_stream, raw=True)
    operation_url = response.headers["Operation-Location"]
    operation_id = operation_url.split("/")[-1]

    while True:
        result = cv_client.get_read_result(operation_id)
        if result.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
            break
        time.sleep(1)

    extracted_lines = []
    if result.status == OperationStatusCodes.succeeded:
        for page in result.analyze_result.read_results:
            for line in page.lines:
                extracted_lines.append(line.text)

    return "\n".join(extracted_lines)


@app.route("/translate", methods=["POST"])
def translate():
    if "image" not in request.files:
        return jsonify({"error": "Image file is missing"}), 400

    image_bytes = request.files["image"].read()
    file_name = request.args.get("nomFichier", "")

    # Détection langue cible
    if file_name.startswith("FR_"):
        target_lang = "français"
    elif file_name.startswith("AR_"):
        target_lang = "arabe"
    else:
        target_lang = "espagnol"  # valeur par défaut

    # OCR Azure
    ocr_text = extract_text_azure(image_bytes)

    # Encodage image
    image_data = base64.b64encode(image_bytes).decode("utf-8")

    # Prompt selon langue
    if target_lang == "arabe":
        prompt_text = f"""
Voici une image d'un document administratif multilingue (français, arabe, anglais).
⚠️ Chaque mot doit être traduit en arabe — aucun mot ou lettre latine ne doit rester.
- Conserve la mise en page exacte.
- Utilise OCR uniquement pour combler les zones floues.
- Respecte le nombre exact de lignes et colonnes des tableaux.
- Cellules vides = <td>&nbsp;</td>.
- Ne retourne que du HTML valide (sans <html> ni <body>).
Texte OCR :
{ocr_text}
"""
        max_tokens = 1200
    else:
        prompt_text = f"""
Voici une image d'un document administratif multilingue (français, arabe, anglais).
⚠️ Traduis tout en {target_lang}, aucun mot d'une autre langue ne doit rester.
- Respecte l'alignement original (titres, paragraphes, signatures, tampons).
- Les tableaux doivent conserver exactement leur nombre de lignes et colonnes.
- Cellules vides = <td>&nbsp;</td>.
- Ne fusionne ni ne supprime de cellules.
- Ne retourne que du HTML valide (sans <html> ni <body>).
Texte OCR :
{ocr_text}
"""
        max_tokens = 2000

    # Appel API
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://test.local",
        "X-Title": "Traduction Document",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "anthropic/claude-3.5-sonnet",
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]
            }
        ]
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=90
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return jsonify({"error": "Erreur API Claude", "details": str(e)}), 500

    if not data.get("choices"):
        return jsonify({"error": "Pas de réponse du modèle", "details": data}), 502

    model_output = data["choices"][0]["message"]["content"].strip()

    # Si sortie déjà HTML → on garde
    if model_output.startswith("<"):
        html_content = model_output
    else:
        html_content = md2html(model_output)

    # Retirer phrases d’intro
    html_content = re.sub(
        r'^\s*<p>(Aquí está|Voici la traduction|Here is the translation).*?</p>\s*',
        '',
        html_content,
        flags=re.IGNORECASE | re.DOTALL
    )

    # 🔹 Nettoyage lettres latines → UNIQUEMENT pour arabe
    if target_lang == "arabe":
        html_content = re.sub(r'[A-Za-z]', '', html_content)

    return jsonify({
        "ocr_text": ocr_text,
        "html": html_content,
        "langue": target_lang
    })


@app.route("/")
def index():
    return "API OK - POST /translate avec image"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
