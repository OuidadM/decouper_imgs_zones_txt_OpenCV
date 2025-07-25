from flask import Flask, request, jsonify
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import openai
import os
import io  
import time

# Initialisation de Flask et des variables d'environnement
app = Flask(__name__)

# Clés API
AZURE_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_VISION_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Client Azure Computer Vision
cv_client = ComputerVisionClient(AZURE_ENDPOINT, CognitiveServicesCredentials(AZURE_KEY))

# Fonction : OCR avec Azure Read API
def extract_text_azure(image_bytes):
    image_stream = io.BytesIO(image_bytes)  # conversion en stream

    response = cv_client.read_in_stream(image=image_stream, language="ar", raw=True)
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


# Fonction : Traduction avec GPT-4o
def translate_text_with_gpt4o(text, target_lang="French"):
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
                    "N'inclus jamais le texte original en arabe. Ne laisse aucun passage non traduit."
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

@app.route("/")
def index():
    return "API OK - voir /votre-endpoint pour utiliser l'API"
@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "Image file is missing"}), 400

    image_bytes = request.files["image"].read()

    # OCR Azure
    extracted_text = extract_text_azure(image_bytes)

    # Traduction GPT-4o
    translated_text = translate_text_with_gpt4o(extracted_text)

    return jsonify({
        "ocr_text": extracted_text,
        "translation": translated_text
    })

# Lancer le serveur
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
