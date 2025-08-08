from flask import Flask, request, jsonify
import requests, base64, os
from markdown2 import markdown as md2html

app = Flask(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

@app.route("/traduire", methods=["POST"])
def translate():
    if "image" not in request.files:
        return jsonify({"error": "Image manquante"}), 400
    
    image_bytes = request.files["image"].read()
    nom = request.args.get("nomFichier", "document")
    langue = "français" if nom.startswith("FR_") else "espagnol"
    bundle_index = int(request.args.get("bundle", "1"))

    # Encodage image
    image_data = base64.b64encode(image_bytes).decode("utf-8")

    # Prompt
    prompt_text = f"""
Voici une image d'un document administratif multilingue (français, arabe, anglais).
Traduis fidèlement tout le contenu visible en {langue}.
Respecte exactement la structure visuelle. Ne saute aucun élément.
Retourne le texte au format Markdown clair et structuré.
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://test.local",
        "X-Title": "Traduction Document",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "anthropic/claude-3.5-sonnet",
        "max_tokens": 800,
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
        return jsonify({"error": "Erreur API", "details": data}), 500

    markdown_text = data["choices"][0]["message"]["content"]

    # Convertir Markdown → HTML (Google Docs Create Document attend du HTML)
    html_content = md2html(markdown_text)

    # Vérifier si c'est le dernier bundle
    is_last = request.args.get("last", "false").lower() == "true"

    return jsonify({
        "status": f"Page {bundle_index} traitée",
        "langue": langue,
        "is_last": is_last,
        "html": html_content
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
