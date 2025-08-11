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
    target_lang = "français" if file_name.startswith("FR_") else "espagnol"

    # 1️⃣ OCR Azure
    ocr_text = extract_text_azure(image_bytes)

    # 2️⃣ Encodage image
    image_data = base64.b64encode(image_bytes).decode("utf-8")

    # 3️⃣ Prompt combiné (strict pour HTML pur)
    prompt_text = f"""
Voici une image d'un document administratif multilingue (français, arabe, anglais).
Traduis fidèlement tout le contenu visible en {target_lang}.
Utilise le texte OCR fourni ci-dessous pour compléter les zones floues ou difficiles à lire.
Reproduis la mise en page et la structure exacte : titres, paragraphes, tableaux, tampons, signatures.
Conserve toutes les colonnes/lignes des tableaux, même vides (utilise <td>&nbsp;</td> si nécessaire).
Ne saute aucun élément.
⚠️ IMPORTANT : Retourne UNIQUEMENT le contenu traduit au format HTML valide.
Pas d'explication, pas de phrase d'introduction, pas de commentaire.
Pas de balises <html>, <head> ou <body>.
Texte OCR extrait :
{ocr_text}
"""

    # 4️⃣ Appel Claude Sonnet 3.5 via OpenRouter
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://test.local",
        "X-Title": "Traduction Document",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "anthropic/claude-3.5-sonnet",
        "max_tokens": 2000,
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

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    data = response.json()

    if "choices" not in data:
        return jsonify({"error": "Erreur API Claude", "details": data}), 500

    model_output = data["choices"][0]["message"]["content"].strip()

    # 5️⃣ Si sortie déjà en HTML → on garde, sinon on convertit Markdown → HTML
    if model_output.startswith("<"):
        html_content = model_output
    else:
        html_content = md2html(model_output)

    # 6️⃣ Suppression auto des phrases d’introduction éventuelles
    html_content = re.sub(
        r'^\s*<p>(Aquí está|Voici la traduction|Here is the translation).*?</p>\s*',
        '',
        html_content,
        flags=re.IGNORECASE | re.DOTALL
    )

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
