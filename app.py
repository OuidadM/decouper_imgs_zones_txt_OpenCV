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
    if file_name.startswith("FR_"):
        target_lang = "français"
    elif file_name.startswith("AR_"):
        target_lang = "arabe"
    else:
        target_lang = "espagnol"  # valeur par défaut


    # 1️⃣ OCR Azure
    ocr_text = extract_text_azure(image_bytes)

    # 2️⃣ Encodage image
    image_data = base64.b64encode(image_bytes).decode("utf-8")

    # 3️⃣ Prompt amélioré : priorité à l'image pour la structure, OCR pour combler
    prompt_text = f"""
Voici une image d'un document administratif multilingue (français, arabe, anglais).
⚠️ Traduis OBLIGATOIREMENT TOUT le texte en {target_lang}, même si le texte est déjà lisible.
Aucun mot ne doit rester dans une autre langue que {target_lang}.

⚠️ Règles obligatoires :
- Utilise en priorité la structure visuelle de l'image pour reproduire la mise en page exacte.
- N'utilise le texte OCR ci-dessous que pour compléter les parties floues ou difficiles à lire, sans casser la structure détectée sur l'image.
- Respecte l'alignement d'origine (gauche, centré, droite) pour titres, paragraphes, signatures, tampons.
- Pour chaque tableau : détecte le nombre exact de lignes et colonnes depuis l'image, puis remplis-le intégralement.
- Inclure toutes les cellules, même vides ou avec des astérisques (*****). Si vide, mets <td>&nbsp;</td>.
- Ne fusionne pas de cellules, ne supprime aucune ligne ou colonne.
- Reproduis aussi tous les autres éléments visuels (mentions marginales, codes, logos, petits caractères).
- Retourne uniquement du HTML valide (<h1>, <h2>, <p>, <table>, <thead>, <tbody>, <tr>, <th>, <td>, <strong>, <em>).
- Aucune phrase d'introduction, aucun commentaire.

Texte OCR (à utiliser uniquement pour combler les zones floues) :
{ocr_text}
- Chaque élément textuel (titres, paragraphes, tableaux) doit être traduit mot à mot en {target_lang}, aucun mot original ne doit subsister.
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
