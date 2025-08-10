from flask import Flask, request, jsonify
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import openai
import requests
import os
import io
import time

# Initialisation Flask
app = Flask(__name__)

# Variables d'environnement
AZURE_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_VISION_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # ta clé OpenRouter
openai.api_key = os.getenv("OPENAI_API_KEY")

# Client Azure Computer Vision
cv_client = ComputerVisionClient(AZURE_ENDPOINT, CognitiveServicesCredentials(AZURE_KEY))

# === OCR avec Azure ===
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

# === Traduction avec Claude Sonnet 3.5 via OpenRouter ===
def translate_text_with_claude(text, target_lang="Spanish"):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://test.local",  # requis par OpenRouter
        "X-Title": "Traduction Document",
        "Content-Type": "application/json"
    }

    prompt_system = (
        "Tu es un traducteur professionnel spécialisé dans la traduction officielle "
        "de documents administratifs et juridiques. Traduis fidèlement et exactement "
        "tout le contenu fourni, en respectant le ton formel et administratif. "
        f"Ta réponse doit être exclusivement en {target_lang}. "
        "Ne simplifie rien, ne reformule rien, ne commente rien. "
        "Conserve la structure logique, les noms propres, les dates, "
        "les références de lois ou de décrets, et tous les termes juridiques. "
        "Évite toute approximation. Si une partie du texte est illisible, indique [illisible]. "
        "Ne conserve aucun texte non traduit. Aucun commentaire ou introduction, uniquement la traduction."
    )

    payload = {
        "model": "anthropic/claude-3.5-sonnet",
        "max_tokens": 2000,
        "messages": [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": text}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()

    if "choices" not in data:
        raise Exception(f"Erreur API Claude: {data}")

    return data["choices"][0]["message"]["content"].strip()

# Fonction : Traduction avec GPT-4o
def translate_text_with_gpt4o(text, target_lang="Spanish"):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un traducteur professionnel spécialisé dans la traduction officielle de documents administratifs et juridiques. "
                    "Traduis fidèlement et exactement tout le contenu fourni, en respectant le ton formel et administratif. "
                    f"Ta réponse doit être exclusivement en {target_lang}. "
                    "Ne simplifie rien, ne reformule rien, ne commente rien. "
                    "Conserve la structure logique, les noms propres, les dates, les références de lois ou de décrets, et tous les termes juridiques ou institutionnels. "
                    "Évite toute approximation. Si une partie du texte est illisible ou incohérente, indique [illisible] sans rien inventer. "
                    "Chaque mot et chaque phrase du texte doivent être traduits. "
                    "Aucun passage, même partiel, ne doit rester dans la langue originale. "
                    "Il est strictement interdit de conserver du texte non traduit. "
                    f"La traduction doit être complète, fidèle et entièrement rédigée en {target_lang}, sans exception. "
                    "Ne commence pas ta réponse par 'Voici la traduction' ni aucune formule explicative. Réponds uniquement avec la traduction directe, sans commentaire ni introduction."
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

# === Endpoint Flask ===
@app.route("/translate", methods=["POST"])
def translate():
    if "image" not in request.files:
        return jsonify({"error": "Image file is missing"}), 400

    image_bytes = request.files["image"].read()
    file_name = request.args.get("nomFichier", "")
    model=request.args.get("modele", "")

    target_lang = "French" if file_name.startswith("FR_") else "Spanish"

    extracted_text = extract_text_azure(image_bytes)
    translated_text = translate_text_with_claude(extracted_text, target_lang) if model.startswith("claude") else translate_text_with_gpt4o(extracted_text, target_lang)

    return jsonify({
        "ocr_text": extracted_text,
        "translation": translated_text,
        "langue": target_lang
    })

@app.route("/")
def index():
    return "API OK - Endpoint /detect pour traiter les documents"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
