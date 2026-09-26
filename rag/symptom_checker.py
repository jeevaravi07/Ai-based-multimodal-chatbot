"""
Second, independent pipeline: a classic symptom -> disease -> description /
precaution lookup. Deliberately separate from the case-report RAG above --
different dataset, different question.

The case-report RAG answers "which past case report reads like this?" (a
narrative match). This answers a more direct question: "what is this
called, and what's the standard advice?" -- exactly what someone typing a
list of symptoms into the chat wants, and this dataset is *built* to have
that as a clean label, which the case-report CSV never had.

Dataset (data/symptom_checker/), unchanged from source:
    Training.csv            132 binary symptom columns + `prognosis` (disease)
                             label, 4920 rows
    Testing.csv              held-out rows in the same format, 42 rows
    symptom_Description.csv  disease -> one-paragraph description, 41 rows
    symptom_precaution.csv   disease -> up to 4 precaution/treatment steps, 41 rows
    Symptom_severity.csv     symptom -> severity weight, 132 rows (not currently
                             used for prediction; kept for anyone who wants to
                             weight symptoms later)
Source: itachi9604/healthcare-chatbot (public GitHub repo, itself a mirror
of the Kaggle "disease-symptom-description-dataset"). Trained locally here,
no API key, no internet needed after the one-time data pull.

Usage:
    train(verbose=True)              # fit + save the classifier (run via build_index.py)
    predict_from_text(user_text)     # -> dict: disease, confidence, matched_symptoms,
                                      #    description, precautions, alternatives
                                      #    or None if no known symptom was recognized
"""
import csv
import re

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import config

_MODEL_CACHE = None
_FEATURES_CACHE = None
_DESCRIPTIONS_CACHE = None
_PRECAUTIONS_CACHE = None


def _canonical_symptom_columns(df: pd.DataFrame) -> list:
    return [c.strip() for c in df.columns if c != "prognosis"]


def _load_two_col_csv(path: str) -> dict:
    """symptom_Description.csv / symptom_precaution.csv ship with NO header
    row: first column is the disease name, the rest are the payload."""
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            disease = row[0].strip()
            rest = [c.strip() for c in row[1:] if c and c.strip()]
            out[disease] = rest[0] if len(rest) == 1 else rest
    return out


def train(verbose: bool = False):
    df = pd.read_csv(config.SYMPTOM_TRAINING_CSV)
    df.columns = [c.strip() for c in df.columns]
    feature_cols = _canonical_symptom_columns(df)

    X = df[feature_cols].values
    y = df["prognosis"].str.strip()

    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    clf.fit(X, y)

    if verbose:
        try:
            test_df = pd.read_csv(config.SYMPTOM_TESTING_CSV)
            test_df.columns = [c.strip() for c in test_df.columns]
            X_test = test_df[feature_cols].values
            y_test = test_df["prognosis"].str.strip()
            acc = accuracy_score(y_test, clf.predict(X_test))
            print(f"Symptom checker held-out accuracy: {acc:.3f} on {len(test_df)} rows")
        except FileNotFoundError:
            pass

    joblib.dump(clf, config.SYMPTOM_CHECKER_MODEL_PATH)
    joblib.dump(feature_cols, config.SYMPTOM_CHECKER_FEATURES_PATH)

    if verbose:
        print(f"Trained on {len(df)} rows, {len(feature_cols)} known symptoms, "
              f"{y.nunique()} diseases. Saved to {config.SYMPTOM_CHECKER_MODEL_PATH}")
    return clf, feature_cols


def _load():
    global _MODEL_CACHE, _FEATURES_CACHE, _DESCRIPTIONS_CACHE, _PRECAUTIONS_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = joblib.load(config.SYMPTOM_CHECKER_MODEL_PATH)
        _FEATURES_CACHE = joblib.load(config.SYMPTOM_CHECKER_FEATURES_PATH)
        _DESCRIPTIONS_CACHE = _load_two_col_csv(config.SYMPTOM_DESCRIPTION_CSV)
        _PRECAUTIONS_CACHE = _load_two_col_csv(config.SYMPTOM_PRECAUTION_CSV)
    return _MODEL_CACHE, _FEATURES_CACHE, _DESCRIPTIONS_CACHE, _PRECAUTIONS_CACHE


_WORD_SYNONYMS = {
    "aching": "pain", "ache": "pain", "achy": "pain", "sore": "pain", "hurts": "pain", "hurting": "pain",
    "itchy": "itch",
    "tired": "fatigue", "exhausted": "fatigue", "sleepy": "fatigue",
    "dizzy": "dizziness",
    "nauseous": "nausea", "queasy": "nausea",
}


_PHRASE_SYNONYMS = {
    "loose motion": "diarrhoea", "loose stool": "diarrhoea", "watery stool": "diarrhoea",
    "throwing up": "vomiting", "throw up": "vomiting", "vomited": "vomiting",
    "stomach ache": "stomach_pain", "stomachache": "stomach_pain",
    "body ache": "muscle_pain", "body pain": "muscle_pain",
    "losing weight": "weight_loss", "gaining weight": "weight_gain",
    "yellow skin": "yellowish_skin",
    "short of breath": "breathlessness", "shortness of breath": "breathlessness",
    "difficulty breathing": "breathlessness",
    "stuffy nose": "congestion", "blocked nose": "congestion",
    "high temperature": "high_fever",
    "red patches": "skin_rash",
}

_STOPWORDS = {"and", "of", "in", "on", "the", "to", "a", "an", "with", "from", "at", "is", "are", "my", "i"}


def _normalize_word(word: str) -> str:
    word = _WORD_SYNONYMS.get(word, word)
    for suffix, min_len in (("ing", 6), ("ed", 5), ("es", 5), ("s", 4)):
        if word.endswith(suffix) and len(word) >= min_len and not word.endswith("ss"):
            word = word[: -len(suffix)]
            break
    return word


def _tokenize(text: str) -> set:
    raw = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    raw = re.sub(r"\s+", " ", raw).strip()
    return {_normalize_word(w) for w in raw.split(" ") if w and w not in _STOPWORDS}, raw


def extract_symptoms_from_text(text: str, feature_cols: list) -> list:
    """Map free-text symptom descriptions onto the dataset's known symptom
    column names. Matches in three passes, most to least strict:
      1. the exact phrase as a substring ('skin rash' typed verbatim)
      2. a curated colloquial phrase ('loose motion' -> diarrhoea)
      3. word-order-independent overlap after light stemming/synonyms
         ('joints are aching with pain' -> all of joint_pain's words
         present, regardless of order or what's in between)
    """
    normalized_input, raw_padded = _tokenize(text)
    padded = f" {raw_padded} "
    matched = []

    for col in feature_cols:
        phrase = col.replace("_", " ").strip()

        if f" {phrase} " in padded:
            matched.append(col)
            continue

        if any(f" {p} " in padded or padded.strip().startswith(p) for p, mapped in
               _PHRASE_SYNONYMS.items() if mapped == col):
            matched.append(col)
            continue

        phrase_words = [w for w in phrase.split(" ") if w and w not in _STOPWORDS]
        significant = {_normalize_word(w) for w in phrase_words}
        if significant and significant.issubset(normalized_input):
            matched.append(col)

    return matched


def predict_from_text(text: str):
    clf, feature_cols, descriptions, precautions = _load()
    matched = extract_symptoms_from_text(text, feature_cols)

    if len(matched) < config.MIN_SYMPTOM_MATCH_COUNT:
        return None

    row = {col: (1 if col in matched else 0) for col in feature_cols}
    X = pd.DataFrame([row])[feature_cols].values

    proba = clf.predict_proba(X)[0]
    classes = clf.classes_
    best_idx = proba.argmax()
    disease = classes[best_idx]
    confidence = float(proba[best_idx])

    top3_idx = proba.argsort()[::-1][:3]
    alternatives = [{"disease": classes[i], "confidence": round(float(proba[i]), 3)}
                    for i in top3_idx if i != best_idx and proba[i] > 0]

    return {
        "disease": disease,
        "confidence": round(confidence, 3),
        "matched_symptoms": [s.replace("_", " ") for s in matched],
        "description": descriptions.get(disease, ""),
        "precautions": precautions.get(disease, []),
        "alternatives": alternatives,
    }


if __name__ == "__main__":
    train(verbose=True)
