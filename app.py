"""
Flask front end for Medical_RAG_Chatbot.

One chat box, three ways to use it:
  - Type symptoms only        -> symptom checker panel (disease + precautions)
                                   AND the closest matching case reports
  - Upload an image only       -> predicted scan type + closest matching case reports
  - Upload an image + symptoms -> all of the above, together

No LLM, no API key -- everything here runs against the locally trained
FAISS indices and classifiers built by scripts/build_index.py.
"""
import os
import uuid

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

import config
from rag import retriever, symptom_checker

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  


def _allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_IMAGE_EXTENSIONS
    )


def _indices_ready() -> bool:
    return os.path.exists(config.TEXT_FAISS_INDEX) and os.path.exists(config.IMAGE_FAISS_INDEX)


@app.route("/")
def index():
    return render_template("index.html", ready=_indices_ready())


@app.route("/chat", methods=["POST"])
def chat():
    if not _indices_ready():
        return jsonify({
            "error": "The indices haven't been built yet. Run `python scripts/build_index.py` first."
        }), 503

    symptom_text = (request.form.get("message") or "").strip()
    image_file = request.files.get("image")

    response = {"symptom_check": None, "case_matches": [], "predicted_modality": None,
                "used_image": False, "used_text": bool(symptom_text)}

    saved_image_path = None
    original_filename = None
    if image_file and image_file.filename and _allowed_file(image_file.filename):
        original_filename = image_file.filename
        ext = original_filename.rsplit(".", 1)[1].lower()
        fname = f"{uuid.uuid4().hex}.{ext}"
        saved_image_path = os.path.join(config.UPLOAD_DIR, fname)
        image_file.save(saved_image_path)
        response["used_image"] = True
        response["uploaded_image_url"] = f"/static/uploads/{fname}"


    if symptom_text:
        try:
            result = symptom_checker.predict_from_text(symptom_text)
            response["symptom_check"] = result
        except FileNotFoundError:
            response["symptom_check"] = None


    if saved_image_path:
        exact_match = retriever.retrieve_by_filename(original_filename)
        if exact_match:

            response["case_matches"] = [exact_match]
            response["predicted_modality"] = exact_match["modality"]
        else:
            case_results, predicted_modality = retriever.retrieve_by_image(saved_image_path)
            response["case_matches"] = case_results
            response["predicted_modality"] = predicted_modality
    elif symptom_text:
        response["case_matches"] = retriever.retrieve_by_text(symptom_text)

    if not symptom_text and not saved_image_path:
        return jsonify({"error": "Please type your symptoms or upload an image."}), 400

    return jsonify(response)


if __name__ == "__main__":
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
        use_reloader=config.FLASK_USE_RELOADER,
    )
