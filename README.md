Medical_RAG_Chatbot — Differential

A fully local, privacy-first healthcare chatbot powered by two independent machine learning pipelines operating behind a single chat interface.

Zero API keys. No external LLMs. 100% on-premise execution.

🧠 System Architecture

The application routes user queries through two distinct, non-overlapping pipelines to ensure accurate, context-appropriate responses:

1. Case-Report RAG (Retrieval-Augmented Generation)

Answers the question: "Which past case report reads like this?"

Input: Text description of symptoms or an uploaded medical image.

Engine: Text is embedded using TF-IDF + SVD. Images use a 512-dimensional handcrafted image descriptor.

Retrieval & Output: Queries are matched against data/cases.csv via a FAISS vector index and summarized using an extractive TextRank summarizer (bypassing the need for generative LLMs).

2. Symptom Checker

Answers the question: "What is this condition called, and what is the standard advice?"

Input: Direct symptom keywords.

Engine: A Random Forest classifier trained on 41 diseases and 132 symptoms (sourced via itachi9604/healthcare-chatbot, mirroring the Kaggle disease-symptom dataset).

Output: Predicts the disease name and provides standard medical descriptions and recommended precautions.

🚀 Getting Started

Prerequisites

Ensure you have Python installed, then install the required dependencies:

pip install -r requirements.txt


Build & Run

Build the Index: Run this one-time setup script to generate the FAISS indices and train the models (run this again anytime your data/ folder changes).

python scripts/build_index.py


Start the Application:

python app.py


Open your browser and navigate to http://127.0.0.1:5000.

📂 Project Structure

Medical_RAG_Chatbot/
├── app.py                      # Flask app — the /chat endpoint
├── config.py                   # Centralized paths and system settings
├── requirements.txt            # Python dependencies
│
├── data/
│   ├── cases.csv               # MultiCaRe-style case reports (image_name, article_id, case_id, case_text)
│   ├── images/                 # Raw case images (331 total)
│   ├── images_sorted/          # Auto-generated: Images sorted by modality (X-ray, CT, MRI, Ultrasound, etc.)
│   └── symptom_checker/        # Training/Testing CSVs, disease descriptions, and precautions
│
├── rag/
│   ├── prepare_data.py         # Splits case_text into CARE sections, detects/sorts scan types
│   ├── text_embeddings.py      # TF-IDF + SVD encoder for case text
│   ├── vector_store.py         # FAISS index build/save/load + metadata handling
│   ├── image_retriever.py      # Handcrafted image descriptor + modality classifier
│   ├── retriever.py            # Core retrieval logic (retrieve_by_text / retrieve_by_image)
│   ├── summarizer.py           # Extractive TextRank summarizer
│   └── symptom_checker.py      # Independent symptom-to-disease RF pipeline
│
├── scripts/
│   └── build_index.py          # Unified script executing all 4 build steps in sequence
│
├── templates/index.html        # Web chat UI
├── static/css/style.css        # UI styling
├── static/uploads/             # Temporary storage for request-time image uploads
├── vector_db/                  # Compiled FAISS indices (cases_text.faiss, cases_image.faiss) and metadata
└── models/                     # Saved models (tfidf_vectorizer, svd_reducer, classifiers)
