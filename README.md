# Medical_RAG_Chatbot — Differential

Two independent local pipelines behind one chat box. No API key, no LLM,
nothing leaves the machine.

1. **Case-report RAG** — upload an image or describe symptoms, get the
   closest matching case reports from `data/cases.csv` (TF-IDF+SVD text
   embeddings / a handcrafted image descriptor, both indexed with FAISS,
   summarized with an extractive TextRank summarizer).
2. **Symptom checker** — type symptoms, get a direct disease name +
   description + precautions, from a RandomForest trained on
   `data/symptom_checker/` (41 diseases, 132 symptoms; source:
   [itachi9604/healthcare-chatbot](https://github.com/itachi9604/healthcare-chatbot),
   itself a mirror of the Kaggle disease-symptom-description dataset).

The two never get confused with each other: the case-report RAG answers
*"which past case report reads like this?"*; the symptom checker answers
*"what is this called, and what's the standard advice?"*.

## Run it

```bash
pip install -r requirements.txt
python scripts/build_index.py   # one-time (or whenever data/ changes)
python app.py
```

Then open `http://127.0.0.1:5000`.

## Project layout

```
Medical_RAG_Chatbot/
├── app.py                     Flask app — the /chat endpoint
├── config.py                  all paths + settings, one place to change them
├── requirements.txt
│
├── data/
│   ├── cases.csv               your MultiCaRe-style case reports (image_name, article_id, case_id, case_text)
│   ├── images/                 the 331 raw case images you provided
│   ├── images_sorted/          auto-generated: same images, sorted into X-ray/CT/MRI/Ultrasound/... folders
│   └── symptom_checker/        Training.csv, Testing.csv, symptom_Description.csv, symptom_precaution.csv, Symptom_severity.csv
│
├── rag/
│   ├── prepare_data.py         splits case_text into CARE sections, detects+sorts scan type
│   ├── text_embeddings.py      TF-IDF + SVD encoder for case text
│   ├── vector_store.py         FAISS index build/save/load + metadata
│   ├── image_retriever.py      handcrafted image descriptor + modality classifier
│   ├── retriever.py            ties the above together: retrieve_by_text / retrieve_by_image
│   ├── summarizer.py           extractive TextRank summarizer (no LLM)
│   └── symptom_checker.py      the independent symptom -> disease pipeline
│
├── scripts/
│   └── build_index.py          one command, runs all 4 build steps in order
│
├── templates/index.html        chat UI
├── static/css/style.css
├── static/uploads/              uploaded images land here at request time
├── vector_db/                  cases_text.faiss, cases_image.faiss, metadata.pkl
└── models/                     tfidf_vectorizer, svd_reducer, modality_classifier,
                                 symptom_checker_classifier + its feature list
```

## Notes / known limits

- `data/cases.csv` has 331 rows across 243 unique cases and no clean
  "diagnosis" column of its own — that's exactly why the symptom checker
  above is a *separate* dataset that does have clean disease labels,
  rather than trying to extract a diagnosis field from free narrative text.
- Scan-type detection (`prepare_data.py`) is keyword-based on the case
  narrative, not a trained image classifier at *build* time — but the
  modality classifier trained in step 3 of the build *does* predict scan
  type from the image itself at request time, which is what the app uses.
- `IMAGE_ENCODER_MODE` is pinned to `"handcrafted"` in `config.py` on
  purpose: it's a 512-dim descriptor with no downloaded weights, so build
  time and request time can never disagree on the vector's dimensionality.
- The symptom checker only recognizes its fixed vocabulary of 132 symptom
  phrases (e.g. `skin_rash`, `high_fever`). Free text outside that
  vocabulary (e.g. "I feel a bit off") returns no match — that's a
  deliberate refusal to guess, not a bug.
