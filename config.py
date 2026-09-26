"""
Central configuration for Medical_RAG_Chatbot.
Two independent pipelines live in this project:

1. Case-report RAG (image OR free text -> similar published case reports)
   Source: MultiCaRe-style case_text + case images you provided
           (data/cases.csv, data/images/).
   Answers: "which past case report reads like this?"

2. Symptom checker (free text symptoms -> a direct disease name)
   Source: itachi9604/healthcare-chatbot (public GitHub repo, itself a
   mirror of the Kaggle "disease-symptom-description-dataset") --
   41 diseases, 132 named symptoms. Trained locally, no API key.
           (data/symptom_checker/*.csv)
   Answers: "what is this called, and what's the standard advice?"

Both run 100% locally -- no LLM, no API key.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------- case-report RAG ---
DATA_DIR = os.path.join(BASE_DIR, "data")
CASES_CSV = os.path.join(DATA_DIR, "cases.csv")
IMAGES_RAW_DIR = os.path.join(DATA_DIR, "images")          
IMAGES_SORTED_DIR = os.path.join(DATA_DIR, "images_sorted")  

VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
TEXT_FAISS_INDEX = os.path.join(VECTOR_DB_DIR, "cases_text.faiss")
IMAGE_FAISS_INDEX = os.path.join(VECTOR_DB_DIR, "cases_image.faiss")
METADATA_PKL = os.path.join(VECTOR_DB_DIR, "metadata.pkl")

MODELS_DIR = os.path.join(BASE_DIR, "models")
TEXT_VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
TEXT_SVD_PATH = os.path.join(MODELS_DIR, "svd_reducer.joblib")
MODALITY_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "modality_classifier.joblib")

TEXT_EMBED_DIM = 128          
IMAGE_EMBED_DIM = 512         
IMAGE_ENCODER_MODE = "handcrafted"  

TOP_K_CASES = 3               

CARE_SECTIONS = [
    "Patient information",
    "Clinical findings",
    "Diagnostic assessment",
    "Therapeutic intervention",
    "Timeline of current episode",
    "Patient perspective",
    "Informed consent",
]

SCAN_TYPES = ["X-ray", "CT", "MRI", "Ultrasound", "Histopathology", "Photograph", "Other"]

# -------------------------------------------------------- symptom checker ---
SYMPTOM_CHECKER_DIR = os.path.join(DATA_DIR, "symptom_checker")
SYMPTOM_TRAINING_CSV = os.path.join(SYMPTOM_CHECKER_DIR, "Training.csv")
SYMPTOM_TESTING_CSV = os.path.join(SYMPTOM_CHECKER_DIR, "Testing.csv")
SYMPTOM_DESCRIPTION_CSV = os.path.join(SYMPTOM_CHECKER_DIR, "symptom_Description.csv")
SYMPTOM_PRECAUTION_CSV = os.path.join(SYMPTOM_CHECKER_DIR, "symptom_precaution.csv")
SYMPTOM_SEVERITY_CSV = os.path.join(SYMPTOM_CHECKER_DIR, "Symptom_severity.csv")

SYMPTOM_CHECKER_MODEL_PATH = os.path.join(MODELS_DIR, "symptom_checker_classifier.joblib")
SYMPTOM_CHECKER_FEATURES_PATH = os.path.join(MODELS_DIR, "symptom_checker_features.joblib")
MIN_SYMPTOM_MATCH_COUNT = 1     

# ------------------------------------------------------------------- misc ---
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True
FLASK_USE_RELOADER = False   
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")

for _d in (VECTOR_DB_DIR, MODELS_DIR, UPLOAD_DIR, IMAGES_SORTED_DIR):
    os.makedirs(_d, exist_ok=True)
