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
        target_lang = "espagnol"

    # OCR Azure
    ocr_text = extract_text_azure(image_bytes)

    # Encodage image
    image_data = base64.b64encode(image_bytes).decode("utf-8")

    # Prompt selon langue (sans HTML)
    if target_lang == "arabe":
        prompt_text = f"""
Voici une image d'un document administratif multilingue (français, arabe, anglais).
⚠️ Chaque mot doit être traduit en arabe — aucun mot ou lettre latine ne doit rester.
- Reproduis exactement la mise en page de l'image, mais en texte brut.
- Utilise les espaces, tabulations et retours à la ligne pour aligner le texte comme sur l'image.
- Conserve tous les éléments visuels : titres, paragraphes, colonnes, tableaux (en texte), signatures, tampons.
- N'ajoute pas de balises HTML, pas de Markdown, pas de codes de formatage.
- Utilise l'OCR ci-dessous uniquement pour combler les zones floues.

Texte OCR :
{ocr_text}
"""
    else:
        prompt_text = f"""
Voici une image d'un document administratif multilingue (français, arabe, anglais).
⚠️ Traduis intégralement en {target_lang}.
- Reproduis exactement la mise en page de l'image, mais en texte brut.
- Utilise les espaces, tabulations et retours à la ligne pour aligner le texte comme sur l'image.
- Conserve tous les éléments visuels : titres, paragraphes, colonnes, tableaux (en texte), signatures, tampons.
- N'ajoute pas de balises HTML, pas de Markdown, pas de codes de formatage.
- Utilise l'OCR ci-dessous uniquement pour combler les zones floues.

Texte OCR :
{ocr_text}
"""

    # Appel API
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://test.local",
        "X-Title": "Traduction Document",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "anthropic/claude-3.5-sonnet",
        "max_tokens": 3000,
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
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return jsonify({"error": "Erreur API Claude", "details": str(e)}), 500

    if not data.get("choices"):
        return jsonify({"error": "Pas de réponse du modèle", "details": data}), 502

    model_output = data["choices"][0]["message"]["content"].strip()

    # 🔹 Pour l'arabe : supprimer toute lettre latine résiduelle
    if target_lang == "arabe":
        model_output = re.sub(r'[A-Za-z]', '', model_output)

    return jsonify({
        "ocr_text": ocr_text,
        "texte": model_output,
        "langue": target_lang
    })
