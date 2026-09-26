"""
Step 2 of the case-report RAG pipeline: turns case narratives (and later,
free-text symptom queries) into fixed-length dense vectors, no API key,
no downloaded language model -- just TF-IDF + SVD (a.k.a. LSA), fit once
on this project's own case texts and reused for every query.
"""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import config


def fit_text_encoder(texts: list):
    """Fit TF-IDF + SVD on the corpus of case texts and persist both."""
    n_components = min(config.TEXT_EMBED_DIM, max(2, len(texts) - 1))

    vectorizer = TfidfVectorizer(
        max_features=20000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = svd.fit_transform(tfidf_matrix)

    joblib.dump(vectorizer, config.TEXT_VECTORIZER_PATH)
    joblib.dump(svd, config.TEXT_SVD_PATH)
    return embeddings


def load_text_encoder():
    vectorizer = joblib.load(config.TEXT_VECTORIZER_PATH)
    svd = joblib.load(config.TEXT_SVD_PATH)
    return vectorizer, svd


def encode_texts(texts: list):
    """Encode new text(s) (e.g. a symptom query) using the already-fit encoder."""
    vectorizer, svd = load_text_encoder()
    tfidf_matrix = vectorizer.transform(texts)
    return svd.transform(tfidf_matrix)
