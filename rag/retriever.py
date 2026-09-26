"""
Step 5: unified retrieval interface used by app.py.
Given either an uploaded image or a free-text query, returns the top-k
most similar case reports (case_id, modality, summary, image path, score).
"""
import os

import config
from rag import text_embeddings, vector_store, image_retriever, summarizer


def retrieve_by_text(query_text: str, top_k: int = config.TOP_K_CASES) -> list:
    records = vector_store.load_metadata()
    index = vector_store.load_index(config.TEXT_FAISS_INDEX)
    query_vec = text_embeddings.encode_texts([query_text])[0]
    distances, indices = vector_store.search(index, query_vec, top_k)
    return _format_results(records, distances, indices)


def retrieve_by_filename(uploaded_filename: str):
    """Exact-match fast path: if the uploaded file's ORIGINAL name matches
    an image_name already in data/cases.csv, skip similarity search
    entirely and return that case's own summary directly. Returns None
    if the name isn't in the dataset (caller should fall back to
    retrieve_by_image for content-based similarity search instead)."""
    if not uploaded_filename:
        return None
    records = vector_store.load_metadata()
    for record in records:
        if record["image_name"] == uploaded_filename:
            return {
                "case_id": record["case_id"],
                "article_id": record["article_id"],
                "image_name": record["image_name"],
                "modality": record["modality"],
                "summary": summarizer.summarize_case(record),
                "distance": 0.0,
                "exact_filename_match": True,
            }
    return None


def retrieve_by_image(image_path: str, top_k: int = config.TOP_K_CASES) -> list:
    records = vector_store.load_metadata()
    index = vector_store.load_index(config.IMAGE_FAISS_INDEX)
    query_vec = image_retriever.encode_image(image_path)
    distances, indices = vector_store.search(index, query_vec, top_k)

    predicted_modality = None
    if os.path.exists(config.MODALITY_CLASSIFIER_PATH):
        try:
            predicted_modality = image_retriever.predict_modality(image_path)
        except Exception:
            predicted_modality = None

    results = _format_results(records, distances, indices)
    return results, predicted_modality


def _format_results(records: list, distances, indices) -> list:
    results = []
    for dist, idx in zip(distances, indices):
        if idx < 0 or idx >= len(records):
            continue
        record = records[idx]
        results.append({
            "case_id": record["case_id"],
            "article_id": record["article_id"],
            "image_name": record["image_name"],
            "modality": record["modality"],
            "summary": summarizer.summarize_case(record),
            "distance": float(dist),
        })
    return results
