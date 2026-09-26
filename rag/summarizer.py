"""
Extractive summarizer for matched case reports -- TextRank over sentences
(networkx PageRank on a sentence-similarity graph). No LLM, no API key.
Always runs on the case's own text; there is nothing to "hallucinate".
"""
import re

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_MIN_LEN_FOR_SUMMARY = 400  # below this, just return the text as-is


def _split_sentences(text: str) -> list:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 3]


def textrank_summarize(text: str, max_sentences: int = 3) -> str:
    sentences = _split_sentences(text)
    if len(text) <= _MIN_LEN_FOR_SUMMARY or len(sentences) <= max_sentences:
        return text.strip()

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf = vectorizer.fit_transform(sentences)
    except ValueError:
        return " ".join(sentences[:max_sentences])

    sim_matrix = (tfidf * tfidf.T).toarray()
    np.fill_diagonal(sim_matrix, 0)

    graph = nx.from_numpy_array(sim_matrix)
    try:
        scores = nx.pagerank(graph, max_iter=200)
    except nx.PowerIterationFailedConvergence:
        scores = {i: 1.0 for i in range(len(sentences))}

    ranked_idx = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    top_idx = sorted(ranked_idx[:max_sentences])  # keep original reading order
    return " ".join(sentences[i] for i in top_idx)


def summarize_case(record: dict, max_sentences: int = 3) -> str:
    """Summarize the most informative section available for a case record
    (falls back through Overview -> Diagnostic assessment -> full text)."""
    sections = record.get("sections", {})
    for key in ("Diagnostic assessment", "Clinical findings", "Overview"):
        if key in sections and sections[key]:
            return textrank_summarize(sections[key], max_sentences)
    return textrank_summarize(record["case_text"], max_sentences)
