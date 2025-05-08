import os
import mimetypes
import pytesseract
import pandas as pd
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from PIL import Image
from werkzeug.utils import secure_filename
import requests
import json
import certifi
import json
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

load_dotenv()

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


CORS(app, resources={r"/*": {"origins": ["*", "https://lovable.so"]}})

# Together API setup (use environment variables or .env)
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_LLM_MODEL = os.getenv("TOGETHER_LLM_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")

# File-type based text extractors
def extract_text_from_pdf(path):
    try:
        reader = PdfReader(path)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    except:
        return ""

def extract_text_from_image(path):
    try:
        return pytesseract.image_to_string(Image.open(path))
    except:
        return ""

def extract_text_from_excel(path):
    try:
        df_list = pd.read_excel(path, sheet_name=None)
        return "\n".join([df.astype(str).to_string(index=False) for df in df_list.values()])
    except:
        return ""

def detect_file_type(path):
    mime, _ = mimetypes.guess_type(path)
    return mime or ""

# LLM-powered classification
def classify_with_llm(doc_text):
    system_prompt = (
        "You are an expert compliance analyst. "
        "You are given the full content of a document (PDF, Excel, or scanned image). "
        "Your job is to classify the document into one of the following categories: "
        "['Sanction Letter', 'Agreement', 'Credit Note', 'Meeting Notes', 'Board resolution', 'Supporting document' 'KYC', 'Invoice', 'Financial Statement', 'Unknown']. "
        "You must also give a confidence score between 0 and 1 and a reason for your classification. "
        "Only return a JSON like: "
        "{\"document_type\": ..., \"confidence\": ..., \"reason\": ...}"
    )

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": TOGETHER_LLM_MODEL,
        "max_tokens": 512,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": doc_text[:6000]}
        ]
    }

    try:
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json=body,
            verify=certifi.where()  # 🚨 disables SSL verification
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return json.dumps({
            "document_type": "Unknown",
            "confidence": 0.0,
            "reason": f"LLM error: {str(e)}"
        })


# Flask route
@app.route("/classify", methods=["POST"])
def classify():
    if "documents" not in request.files:
        return jsonify({"error": "No files uploaded with key 'documents'"}), 400

    files = request.files.getlist("documents")
    results = []

    for file in files:
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        mime = detect_file_type(path)
        text = ""

        if "pdf" in mime:
            text = extract_text_from_pdf(path)
        elif "image" in mime:
            text = extract_text_from_image(path)
        elif "excel" in mime or filename.endswith((".xls", ".xlsx")):
            text = extract_text_from_excel(path)
        else:
            results.append({
                "file_name": filename,
                "document_type": "Unknown",
                "confidence": 0.0,
                "reason": "Unsupported file type"
            })
            continue

        llm_response = classify_with_llm(text)

        try:
            parsed = json.loads(llm_response)
        except:
            parsed = {
                "document_type": "Unknown",
                "confidence": 0.0,
                "reason": f"Invalid JSON from LLM: {llm_response}"
            }

        parsed["file_name"] = filename
        results.append(parsed)

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
