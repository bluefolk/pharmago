from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib
import json
from deep_translator import GoogleTranslator

# 🔄 Load model dan vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = tf.keras.models.load_model("model_diagnosis")

with open("labels.txt", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

with open("penyakit.json", encoding="utf-8") as f:
    penyakit_map = json.load(f)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_text = data.get("text", "").strip()

    if not input_text:
        return jsonify({"error": "Text is required"}), 400

    # ✅ 1. Translate Bahasa Indonesia → English
    translated = GoogleTranslator(source='auto', target='en').translate(input_text)
    print(f"🔁 Translate: '{input_text}' → '{translated}'")

    # ✅ 2. TF-IDF + predict
    X = vectorizer.transform([translated]).toarray()
    y_pred = model.predict(X)[0]
    max_idx = int(np.argmax(y_pred))
    confidence = float(y_pred[max_idx])
    label = labels[max_idx]

    info = penyakit_map.get(label, {})

    return jsonify({
        "penyakit": label,
        "confidence": round(confidence, 4),
        "obat":  info.get("obat", []),
        "saran": translate_en_to_id(info.get("saran", "")),
        "deskripsi": info.get("deskripsi", ""),
        "efek_samping": translate_en_to_id(info.get("efek_samping", ""))
    })


def translate_en_to_id(text):
    try:
        return GoogleTranslator(source='en', target='id').translate(text)
    except:
        return text  # fallback if error
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
